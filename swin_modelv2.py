import os
from collections import defaultdict, OrderedDict

import torch.nn as nn

from utils.parse_config import *
from utils.utils import *
import time
import math

from swin_t.swin_transformer import *

try:
    from utils.syncbn import SyncBN

    batch_norm = SyncBN  # nn.BatchNorm2d
except ImportError:
    batch_norm = nn.BatchNorm2d


def create_modules(module_defs):
    """
    Constructs module list of layer blocks from module configuration in module_defs
    """
    hyperparams = module_defs.pop(0)
    output_filters = [int(hyperparams['channels'])]  # = [3]
    module_list = nn.ModuleList()
    yolo_layer_count = 0
    for i, module_def in enumerate(module_defs):
        modules = nn.Sequential()

        if module_def['type'] == 'convolutional':
            bn = int(module_def['batch_normalize'])
            filters = int(module_def['filters'])
            kernel_size = int(module_def['size'])
            pad = (kernel_size - 1) // 2 if int(module_def['pad']) else 0
            modules.add_module('conv_%d' % i, nn.Conv2d(in_channels=output_filters[-1],
                                                        out_channels=filters,
                                                        kernel_size=kernel_size,
                                                        stride=int(module_def['stride']),
                                                        padding=pad,
                                                        bias=not bn))
            if bn:
                after_bn = batch_norm(filters)
                modules.add_module('batch_norm_%d' % i, after_bn)
                # BN is uniformly initialized by default in pytorch 1.0.1. 
                # In pytorch>1.2.0, BN weights are initialized with constant 1,
                # but we find with the uniform initialization the model converges faster.
                nn.init.uniform_(after_bn.weight)
                nn.init.zeros_(after_bn.bias)
            if module_def['activation'] == 'leaky':
                modules.add_module('leaky_%d' % i, nn.LeakyReLU(0.1))

            output_filters.append(filters)

        elif module_def['type'] == 'patch_embedding':  # swin第一个阶段: patch embedding
            norm_layer = nn.LayerNorm if module_def['norm_layer'] == 'true' else None
            modules.add_module('patch embedding%d' % i, PatchEmbed(patch_size=int(module_def['patch_size']),
                                                                   in_chans=int(module_def['in_channels']),
                                                                   embed_dim=int(module_def['embed_dim']),
                                                                   norm_layer=norm_layer))

            filters = int(module_def['embed_dim'])

            output_filters.append(filters)

        elif module_def['type'] == 'basic_layer':
            downsample = True if module_def['downsample'] == 'true' else False

            modules.add_module('swinT layer%d' % i,
                               BasicLayer(
                                   dim=int(module_def['dim']),
                                   depth=int(module_def['depth']),
                                   num_heads=int(module_def['num_heads']),
                                   window_size=int(module_def['window_size']),
                                   downsample=downsample
                               ))
            # filters = int(module_def['dim'])  # channels doubled after a block
        
        elif module_def['type'] == 'layer_norm':
            modules.add_module('layer norm%d'% i, 
                                nn.LayerNorm(normalized_shape=int(module_def['features'])))

            filters = int(module_def['features'])

            output_filters.append(filters)

        elif module_def['type'] == 'maxpool':
            kernel_size = int(module_def['size'])
            stride = int(module_def['stride'])
            if kernel_size == 2 and stride == 1:
                modules.add_module('_debug_padding_%d' % i, nn.ZeroPad2d((0, 1, 0, 1)))
            maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=int((kernel_size - 1) // 2))
            modules.add_module('maxpool_%d' % i, maxpool)

            output_filters.append(filters)

        elif module_def['type'] == 'upsample':
            upsample = Upsample(scale_factor=int(module_def['stride']))
            modules.add_module('upsample_%d' % i, upsample)

            output_filters.append(filters)

        elif module_def['type'] == 'route':
            layers = [int(x) for x in module_def['layers'].split(',')]  # [-3] or [-3, -1] or [-1, 61] etc
            filters = sum([output_filters[i + 1 if i > 0 else i] for i in layers])
            modules.add_module('route_%d' % i, EmptyLayer())

            output_filters.append(filters)

        elif module_def['type'] == 'shortcut':
            filters = output_filters[int(module_def['from'])]
            modules.add_module('shortcut_%d' % i, EmptyLayer())

            output_filters.append(filters)

        elif module_def['type'] == 'yolo':
            anchor_idxs = [int(x) for x in module_def['mask'].split(',')]
            # Extract anchors
            anchors = [float(x) for x in module_def['anchors'].split(',')]
            anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in anchor_idxs]  # eg [(8,24), (11,34), (16,48), (23,68),]
            nC = int(module_def['classes'])  # number of classes  = 1
            img_size = (int(hyperparams['width']), int(hyperparams['height']))
            # Define detection layer
            yolo_layer = YOLOLayer(anchors, nC, int(hyperparams['nID']),
                                   int(hyperparams['embedding_dim']), img_size, yolo_layer_count)
            modules.add_module('yolo_%d' % i, yolo_layer)
            yolo_layer_count += 1

            output_filters.append(filters)

        # Register module list and number of output filters
        module_list.append(modules)
        # output_filters.append(filters)

    # print(output_filters)
    # exit(0)

    return hyperparams, module_list


class EmptyLayer(nn.Module):
    """Placeholder for 'route' and 'shortcut' layers"""

    def __init__(self):
        super(EmptyLayer, self).__init__()

    def forward(self, x):
        return x


class Upsample(nn.Module):
    # Custom Upsample layer (nn.Upsample gives deprecated warning message)

    def __init__(self, scale_factor=1, mode='nearest'):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)


class YOLOLayer(nn.Module):
    def __init__(self, anchors, nC, nID, nE, img_size, yolo_layer):
        super(YOLOLayer, self).__init__()
        self.layer = yolo_layer
        nA = len(anchors)
        self.anchors = torch.FloatTensor(anchors)
        self.nA = nA  # number of anchors (3)  4
        self.nC = nC  # number of classes (80)  1
        self.nID = nID  # number of identities
        self.img_size = 0
        self.emb_dim = nE
        self.shift = [1, 3, 5]

        self.SmoothL1Loss = nn.SmoothL1Loss()
        self.SoftmaxLoss = nn.CrossEntropyLoss(ignore_index=-1)
        self.CrossEntropyLoss = nn.CrossEntropyLoss()
        self.IDLoss = nn.CrossEntropyLoss(ignore_index=-1)
        self.s_c = nn.Parameter(-4.15 * torch.ones(1))  # -4.15
        self.s_r = nn.Parameter(-4.85 * torch.ones(1))  # -4.85
        self.s_id = nn.Parameter(-2.3 * torch.ones(1))  # -2.3

        self.emb_scale = math.sqrt(2) * math.log(self.nID - 1) if self.nID > 1 else 1

    def forward(self, p_cat, img_size, targets=None, classifier=None, test_emb=False):
        p, p_emb = p_cat[:, :24, ...], p_cat[:, 24:, ...]  # 在channel维度以24为界分开
        nB, nGh, nGw = p.shape[0], p.shape[-2], p.shape[-1]  # bs, h, w

        if self.img_size != img_size:  # True: 0 != [608, 1088]
            create_grids(self, img_size, nGh, nGw)

            if p.is_cuda:
                self.grid_xy = self.grid_xy.cuda()
                self.anchor_wh = self.anchor_wh.cuda()

        p = p.view(nB, self.nA, self.nC + 5, nGh, nGw).permute(0, 1, 3, 4, 2).contiguous()  # prediction
        # p: bs, 4, 1+5, h, w -(permute)-> bs, 4, h, w, 6
        # 5 应该是边界框 + 置信度

        p_emb = p_emb.permute(0, 2, 3, 1).contiguous()
        p_box = p[..., :4]  # 最后一维0~3 bbox
        p_conf = p[..., 4:6].permute(0, 4, 1, 2, 3)  # Conf 最后一维4~5 应该是类别+置信度

        # Training
        if targets is not None:
            if test_emb:
                tconf, tbox, tids = build_targets_max(targets, self.anchor_vec.cuda(), self.nA, self.nC, nGh, nGw)
            else:
                tconf, tbox, tids = build_targets_thres(targets, self.anchor_vec.cuda(), self.nA, self.nC, nGh, nGw)
            tconf, tbox, tids = tconf.cuda(), tbox.cuda(), tids.cuda()
            mask = tconf > 0

            # Compute losses
            nT = sum([len(x) for x in targets])  # number of targets
            nM = mask.sum().float()  # number of anchors (assigned to targets)
            nP = torch.ones_like(mask).sum().float()
            if nM > 0:
                lbox = self.SmoothL1Loss(p_box[mask], tbox[mask])
            else:
                FT = torch.cuda.FloatTensor if p_conf.is_cuda else torch.FloatTensor
                lbox, lconf = FT([0]), FT([0])
            lconf = self.SoftmaxLoss(p_conf, tconf)
            lid = torch.Tensor(1).fill_(0).squeeze().cuda()
            emb_mask, _ = mask.max(1)

            # For convenience we use max(1) to decide the id, TODO: more reseanable strategy
            tids, _ = tids.max(1)
            tids = tids[emb_mask]
            embedding = p_emb[emb_mask].contiguous()
            embedding = self.emb_scale * F.normalize(embedding)
            nI = emb_mask.sum().float()

            if test_emb:
                if np.prod(embedding.shape) == 0 or np.prod(tids.shape) == 0:
                    return torch.zeros(0, self.emb_dim + 1).cuda()
                emb_and_gt = torch.cat([embedding, tids.float()], dim=1)
                return emb_and_gt

            if len(embedding) > 1:
                logits = classifier(embedding).contiguous()
                lid = self.IDLoss(logits, tids.squeeze())

            # Sum loss components
            loss = torch.exp(-self.s_r) * lbox + torch.exp(-self.s_c) * lconf + torch.exp(-self.s_id) * lid + \
                   (self.s_r + self.s_c + self.s_id)
            loss *= 0.5

            return loss, loss.item(), lbox.item(), lconf.item(), lid.item(), nT

        else:
            p_conf = torch.softmax(p_conf, dim=1)[:, 1, ...].unsqueeze(-1)
            p_emb = F.normalize(p_emb.unsqueeze(1).repeat(1, self.nA, 1, 1, 1).contiguous(), dim=-1)
            # p_emb_up = F.normalize(shift_tensor_vertically(p_emb, -self.shift[self.layer]), dim=-1)
            # p_emb_down = F.normalize(shift_tensor_vertically(p_emb, self.shift[self.layer]), dim=-1)
            p_cls = torch.zeros(nB, self.nA, nGh, nGw, 1).cuda()  # Temp
            p = torch.cat([p_box, p_conf, p_cls, p_emb], dim=-1)
            # p = torch.cat([p_box, p_conf, p_cls, p_emb, p_emb_up, p_emb_down], dim=-1)
            p[..., :4] = decode_delta_map(p[..., :4], self.anchor_vec.to(p))
            p[..., :4] *= self.stride

            return p.view(nB, -1, p.shape[-1])


class Darknet(nn.Module):
    """YOLOv3 object detection model"""

    def __init__(self, cfg_dict, nID=0, test_emb=False):
        super(Darknet, self).__init__()
        if isinstance(cfg_dict, str):
            cfg_dict = parse_model_cfg(cfg_dict)
        self.module_defs = cfg_dict
        self.module_defs[0]['nID'] = nID
        self.img_size = [int(self.module_defs[0]['width']), int(self.module_defs[0]['height'])]
        self.emb_dim = int(self.module_defs[0]['embedding_dim'])
        self.hyperparams, self.module_list = create_modules(self.module_defs)
        self.loss_names = ['loss', 'box', 'conf', 'id', 'nT']
        self.losses = OrderedDict()
        for ln in self.loss_names:
            self.losses[ln] = 0
        self.test_emb = test_emb

        self.classifier = nn.Linear(self.emb_dim, nID) if nID > 0 else None

    def forward(self, x, targets=None, targets_len=None):
        # print(x.shape)
        self.losses = OrderedDict()
        for ln in self.loss_names:
            self.losses[ln] = 0
        is_training = (targets is not None) and (not self.test_emb)
        # img_size = x.shape[-1]
        layer_outputs = []
        output = []

        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            mtype = module_def['type']
            if mtype in ['convolutional', 'upsample', 'maxpool']:
                x = module(x)
                # print(f"{mtype}****{x.shape}")
            elif mtype == 'route':
                layer_i = [int(x) for x in module_def['layers'].split(',')]
                if len(layer_i) == 1:
                    x = layer_outputs[layer_i[0]]
                else:
                    x = torch.cat([layer_outputs[i] for i in layer_i], 1)
                # print(f"{mtype}****{x.shape}")
            elif mtype == 'shortcut':
                layer_i = int(module_def['from'])
                x = layer_outputs[-1] + layer_outputs[layer_i]
                # print(f"{mtype}****{x.shape}")
            elif mtype == 'yolo':
                # print(module[0])
                if is_training:  # get loss
                    targets = [targets[i][:int(l)] for i, l in enumerate(targets_len)]
                    x, *losses = module[0](x, self.img_size, targets, self.classifier)
                    for name, loss in zip(self.loss_names, losses):
                        self.losses[name] += loss
                elif self.test_emb:
                    if targets is not None:
                        targets = [targets[i][:int(l)] for i, l in enumerate(targets_len)]
                    x = module[0](x, self.img_size, targets, self.classifier, self.test_emb)
                else:  # get detections
                    x = module[0](x, self.img_size)

                output.append(x)
                # print(f"{mtype}****{x.shape}")
            layer_outputs.append(x)
        # exit(0)
        if is_training:
            self.losses['nT'] /= 3
            output = [o.squeeze() for o in output]
            return sum(output), torch.Tensor(list(self.losses.values())).cuda()
        elif self.test_emb:
            return torch.cat(output, 0)
        return torch.cat(output, 1)


class Swin_JDE(nn.Module):
    """
    YOLOv3 object detection model
    with Swin-T backbone  
    """

    def __init__(self, cfg_dict, nID=0, test_emb=False):
        super(Swin_JDE, self).__init__()
        if isinstance(cfg_dict, str):
            cfg_dict = parse_model_cfg(cfg_dict)
        self.module_defs = cfg_dict
        self.module_defs[0]['nID'] = nID
        self.img_size = [int(self.module_defs[0]['width']), int(self.module_defs[0]['height'])]
        self.emb_dim = int(self.module_defs[0]['embedding_dim'])
        self.hyperparams, self.module_list = create_modules(self.module_defs)
        self.loss_names = ['loss', 'box', 'conf', 'id', 'nT']
        self.losses = OrderedDict()
        for ln in self.loss_names:
            self.losses[ln] = 0
        self.test_emb = test_emb

        self.classifier = nn.Linear(self.emb_dim, nID) if nID > 0 else None

        self.num_features = [96, 96, 96, 192, 192, 384, 384, 768, 768]  # 用于Swin-t的block 输出reshape

    def forward(self, x, targets=None, targets_len=None):
        # print(x.shape)
        self.losses = OrderedDict()
        for ln in self.loss_names:
            self.losses[ln] = 0
        is_training = (targets is not None) and (not self.test_emb)
        # img_size = x.shape[-1]
        layer_outputs = []
        output = []
        # print(self.module_list)
        # exit(0)
        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            mtype = module_def['type']
            if mtype in ['convolutional', 'upsample', 'maxpool']:
                x = module(x)
                # print(f"{mtype}****{x.shape}")
                layer_outputs.append(x)

            elif mtype == 'patch_embedding':
                x = module(x)
                # print(f"{mtype}****{x.shape}")
                layer_outputs.append(x)

            elif mtype == 'basic_layer':  # swin block

                if i == 1:  # 第一个block块 从四维展为三维
                    Wh, Ww = x.size(2), x.size(3)
                    x = x.flatten(2).transpose(1, 2)
                x_out, H, W, x, Wh, Ww = module[0](x, Wh, Ww)  # module[0]??
                # print(f"{mtype}****{x_out.shape}")
                # layer_outputs.append(out)

            elif mtype == 'layer_norm':  # LayerNorm层
                x_out = module(x_out)

                out = x_out.view(-1, H, W, self.num_features[i]).permute(0, 3, 1, 2).contiguous()
                # print(f"{mtype}****{out.shape}")
                layer_outputs.append(out)  # Layernorm的输出才加入

                if i == 8: # last norm layer update x
                    x = out

            elif mtype == 'route':
                layer_i = [int(x) for x in module_def['layers'].split(',')]
                if len(layer_i) == 1:
                    x = layer_outputs[layer_i[0]]
                else:
                    x = torch.cat([layer_outputs[i] for i in layer_i], 1)

                layer_outputs.append(x)
                # print(f"{mtype}****{x.shape}")
            elif mtype == 'shortcut':
                layer_i = int(module_def['from'])
                x = layer_outputs[-1] + layer_outputs[layer_i]
                layer_outputs.append(x)
                # print(f"{mtype}****{x.shape}")
            elif mtype == 'yolo':
                if is_training:  # get loss
                    targets = [targets[i][:int(l)] for i, l in enumerate(targets_len)]
                    x, *losses = module[0](x, self.img_size, targets, self.classifier)
                    for name, loss in zip(self.loss_names, losses):
                        self.losses[name] += loss
                elif self.test_emb:
                    if targets is not None:
                        targets = [targets[i][:int(l)] for i, l in enumerate(targets_len)]
                    x = module[0](x, self.img_size, targets, self.classifier, self.test_emb)
                else:  # get detections
                    x = module[0](x, self.img_size)
                output.append(x)
                layer_outputs.append(x)
                # print(f"{mtype}****{x.shape}")
            # layer_outputs.append(x)
        # exit(0)
        if is_training:
            self.losses['nT'] /= 3
            output = [o.squeeze() for o in output]
            return sum(output), torch.Tensor(list(self.losses.values())).cuda()
        elif self.test_emb:
            return torch.cat(output, 0)
        return torch.cat(output, 1)


def shift_tensor_vertically(t, delta):
    # t should be a 5-D tensor (nB, nA, nH, nW, nC)
    res = torch.zeros_like(t)
    if delta >= 0:
        res[:, :, :-delta, :, :] = t[:, :, delta:, :, :]
    else:
        res[:, :, -delta:, :, :] = t[:, :, :delta, :, :]
    return res


def create_grids(self, img_size, nGh, nGw):
    self.stride = img_size[0] / nGw
    assert self.stride == img_size[1] / nGh, \
        "{} v.s. {}/{}".format(self.stride, img_size[1], nGh)

    # build xy offsets
    grid_x = torch.arange(nGw).repeat((nGh, 1)).view((1, 1, nGh, nGw)).float()
    grid_y = torch.arange(nGh).repeat((nGw, 1)).transpose(0, 1).view((1, 1, nGh, nGw)).float()
    # grid_y = grid_x.permute(0, 1, 3, 2)
    self.grid_xy = torch.stack((grid_x, grid_y), 4)

    # build wh gains
    self.anchor_vec = self.anchors / self.stride
    self.anchor_wh = self.anchor_vec.view(1, self.nA, 1, 1, 2)


def load_darknet_weights(self, weights, cutoff=-1):
    # Parses and loads the weights stored in 'weights'
    # cutoff: save layers between 0 and cutoff (if cutoff = -1 all are saved)
    weights_file = weights.split(os.sep)[-1]

    # Try to download weights if not available locally
    if not os.path.isfile(weights):
        try:
            os.system('wget https://pjreddie.com/media/files/' + weights_file + ' -O ' + weights)
        except IOError:
            print(weights + ' not found')

    # Establish cutoffs
    if weights_file == 'darknet53.conv.74':
        cutoff = 75
    elif weights_file == 'yolov3-tiny.conv.15':
        cutoff = 15

    # Open the weights file
    fp = open(weights, 'rb')
    header = np.fromfile(fp, dtype=np.int32, count=5)  # First five are header values

    # Needed to write header when saving weights
    self.header_info = header

    self.seen = header[3]  # number of images seen during training
    weights = np.fromfile(fp, dtype=np.float32)  # The rest are weights
    fp.close()

    ptr = 0
    for i, (module_def, module) in enumerate(zip(self.module_defs[:cutoff], self.module_list[:cutoff])):
        if module_def['type'] == 'convolutional':
            conv_layer = module[0]
            if module_def['batch_normalize']:
                # Load BN bias, weights, running mean and running variance
                bn_layer = module[1]
                num_b = bn_layer.bias.numel()  # Number of biases
                # Bias
                bn_b = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.bias)
                bn_layer.bias.data.copy_(bn_b)
                ptr += num_b
                # Weight
                bn_w = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.weight)
                bn_layer.weight.data.copy_(bn_w)
                ptr += num_b
                # Running Mean
                bn_rm = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.running_mean)
                bn_layer.running_mean.data.copy_(bn_rm)
                ptr += num_b
                # Running Var
                bn_rv = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.running_var)
                bn_layer.running_var.data.copy_(bn_rv)
                ptr += num_b
            else:
                # Load conv. bias
                num_b = conv_layer.bias.numel()
                conv_b = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(conv_layer.bias)
                conv_layer.bias.data.copy_(conv_b)
                ptr += num_b
            # Load conv. weights
            num_w = conv_layer.weight.numel()
            conv_w = torch.from_numpy(weights[ptr:ptr + num_w]).view_as(conv_layer.weight)
            conv_layer.weight.data.copy_(conv_w)
            ptr += num_w


def load_swin_weights(self, weights):
    """
    加载swin-t 预训练模型
    self: model
    weights: pth file
    """
    # 基本遵循加载DarkNet预训练权重的形式
    check_point = torch.load('weights/swin_t.pth', map_location='cpu')  # 加载模型
    check_point_state_dict = check_point['state_dict']  # 加载权重

    # resume_state_dict = {}  # 存储要加载的权重
    cnt_basic_layer = 0  # 用以给block块计数
    cnt_layer_norm = 0  # 用以给layer_norm计数


    for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
        # i: 序数 module_def: 字典 里面存储模型参数 module: 模型
        if module_def['type'] == 'patch_embedding':  # patch embedding层
            print("loading patch embedding layer")
            curr_module = module[0]
            curr_module_state_dict = curr_module.state_dict()  # 初始化的state dict
            prefix = 'backbone.patch_embed.'  # 在pth文件里的键前缀

            for k in curr_module_state_dict.keys():  # 遍历其中的keys
                curr_module_state_dict[k] = check_point_state_dict[prefix + k]

            curr_module.load_state_dict(curr_module_state_dict)  # 加载进去

        elif module_def['type'] == 'basic_layer':
            print(f"loading basic_layer{cnt_basic_layer}")
            # swin T 的block
            curr_module = module[0]
            curr_module_state_dict = curr_module.state_dict()  # 初始化的state dict
            prefix = 'backbone.layers.' + str(cnt_basic_layer) +'.'

            for k in curr_module_state_dict.keys():
                curr_module_state_dict[k] = check_point_state_dict[prefix + k]

            curr_module.load_state_dict(curr_module_state_dict)  # 加载进去

            cnt_basic_layer += 1

        elif module_def['type'] == 'layer_norm':
            print(f"loading layer_norm{cnt_layer_norm}")
            # block后的layer_norm层
            curr_module = module[0]
            curr_module_state_dict = curr_module.state_dict()  # 初始化的state dict
            prefix = 'backbone.norm' + str(cnt_layer_norm) + '.'

            for k in curr_module_state_dict.keys():
                curr_module_state_dict[k] = check_point_state_dict[prefix + k]

            curr_module.load_state_dict(curr_module_state_dict)  # 加载进去

            cnt_layer_norm += 1

        else:
            break



"""
    @:param path    - path of the new weights file
    @:param cutoff  - save layers between 0 and cutoff (cutoff = -1 -> all are saved)
"""


def save_weights(self, path, cutoff=-1):
    fp = open(path, 'wb')
    self.header_info[3] = self.seen  # number of images seen during training
    self.header_info.tofile(fp)

    # Iterate through layers
    for i, (module_def, module) in enumerate(zip(self.module_defs[:cutoff], self.module_list[:cutoff])):
        if module_def['type'] == 'convolutional':
            conv_layer = module[0]
            # If batch norm, load bn first
            if module_def['batch_normalize']:
                bn_layer = module[1]
                bn_layer.bias.data.cpu().numpy().tofile(fp)
                bn_layer.weight.data.cpu().numpy().tofile(fp)
                bn_layer.running_mean.data.cpu().numpy().tofile(fp)
                bn_layer.running_var.data.cpu().numpy().tofile(fp)
            # Load conv bias
            else:
                conv_layer.bias.data.cpu().numpy().tofile(fp)
            # Load conv weights
            conv_layer.weight.data.cpu().numpy().tofile(fp)

    fp.close()
