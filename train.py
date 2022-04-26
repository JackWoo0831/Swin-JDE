import argparse
# import imp
import json
from statistics import mode
import time
from time import gmtime, strftime
from typing_extensions import final

import torch.optim

import test
# from models import *
# from swin_model import *
from swin_modelv2 import *

from shutil import copyfile
from utils.datasets import JointDataset, collate_fn
from utils.utils import *
from utils.log import logger
from torchvision.transforms import transforms as T


def train(
        cfg,
        data_cfg,
        weights_from="",
        weights_to="",
        save_every=10,
        img_size=(1088, 608),
        resume=False,
        epochs=100,
        batch_size=16,
        accumulated_batches=1,
        freeze_backbone=False,
        opt=None,
):
    # The function starts

    print('args:')
    print(opt)

    final_result = []  # 存储最终结果

    timme = strftime("%Y-%d-%m %H:%M:%S", gmtime())
    timme = timme[5:-3].replace('-', '_')
    timme = timme.replace(' ', '_')
    timme = timme.replace(':', '_')
    weights_to = osp.join(weights_to, 'run' + timme)
    mkdir_if_missing(weights_to)
    if resume:
        latest_resume = osp.join(weights_from, 'latest.pt')

    torch.backends.cudnn.benchmark = True  # unsuitable for multiscale

    # Configure run
    f = open(data_cfg)
    data_config = json.load(f)
    trainset_paths = data_config['train']
    dataset_root = data_config['root']
    f.close()

    transforms = T.Compose([T.ToTensor()])
    # Get dataloader
    dataset = JointDataset(dataset_root, trainset_paths, img_size, augment=True, transforms=transforms)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True,
                                             num_workers=8, pin_memory=True, drop_last=True, collate_fn=collate_fn)
    # Initialize model
    if opt.backbone == 'darknet':
        model = Darknet(cfg, dataset.nID)
    elif opt.backbone == 'swin':
        # model = Swin_Darknet(cfg, dataset.nID)
        model = Swin_JDE(cfg, dataset.nID)
    else:
        raise NotImplementedError

    cutoff = -1  # backbone reaches to cutoff layer
    start_epoch = 1

    if opt.eval_only:  # Edited
        mAP, R, P = test.test(cfg, data_cfg, weights='./weights/latest.pt', batch_size=batch_size, print_interval=40)
        return

    if resume:
        checkpoint = torch.load(latest_resume, map_location='cpu')

        # Load weights to resume from
        model.load_sltate_dict(checkpoint['model'])
        model.cuda().train()

        # Set optimizer
        # optimizer = torch.optim.SGD(filter(lambda x: x.requires_grad, model.parameters()), lr=opt.lr, momentum=.9)
        optimizer = torch.optim.AdamW(filter(lambda x: x.requires_grad, model.parameters()), lr=opt.lr)

        start_epoch = checkpoint['epoch'] + 1
        if checkpoint['optimizer'] is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])

        del checkpoint  # current, saved

    else:
        # Initialize model with backbone (optional)
        # if cfg.endswith('yolov3.cfg'):
        if 'fullswin' in cfg:  # 是完整swin-T的模型
            print("loading swin weights....")
            load_swin_weights(model, osp.join(weights_from, 'swin_t.pth'))
            print("Done!")
        elif 'yolov3' in cfg and 'swin' not in cfg:  # 是DarkNet模型
            print("loading darknet weights....")
            load_darknet_weights(model, osp.join(weights_from, 'darknet53.conv.74'))
            cutoff = 75
            print("done!")
        elif cfg.endswith('yolov3-tiny.cfg'):
            load_darknet_weights(model, osp.join(weights_from, 'yolov3-tiny.conv.15'))
            cutoff = 15
        else:
            print("Attention! No pretrained model loaded")

        model.cuda().train()

        # Set optimizer
        # optimizer = torch.optim.SGD(filter(lambda x: x.requires_grad, model.parameters()), lr=opt.lr, momentum=.9,
        #                             weight_decay=1e-4)
        optimizer = torch.optim.AdamW(filter(lambda x: x.requires_grad, model.parameters()), lr=opt.lr,
                                      weight_decay=0.01)

    model = torch.nn.DataParallel(model)
    # Set scheduler
    """
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                     milestones=[int(0.5 * opt.epochs), int(0.75 * opt.epochs)],
                                                     gamma=0.1)
    """
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                     milestones=[int(opt.epochs - 9), int(opt.epochs - 3)],
                                                     gamma=0.1)

    # An important trick for detection: freeze bn during fine-tuning
    if not opt.unfreeze_bn:
        for i, (name, p) in enumerate(model.named_parameters()):
            p.requires_grad = False if 'batch_norm' in name else True

    # model_info(model)
    t0 = time.time()
    sum_epochs = start_epoch - 1 + epochs

    best_mAP, best_Recall = -1, -1  # use to save the best model

    for epoch in range(epochs):
        epoch += start_epoch
        logger.info(('%8s%12s' + '%10s' * 6) % (
            'Epoch', 'Batch', 'box', 'conf', 'id', 'total', 'nTargets', 'time'))

        # Freeze darknet53.conv.74 for first epoch
        if freeze_backbone and (epoch < 2):
            for i, (name, p) in enumerate(model.named_parameters()):
                if int(name.split('.')[2]) < cutoff:  # if layer < 75
                    p.requires_grad = False if (epoch == 0) else True

        ui = -1
        rloss = defaultdict(float)  # running loss
        optimizer.zero_grad()
        for i, (imgs, targets, _, _, targets_len) in enumerate(dataloader):
            if sum([len(x) for x in targets]) < 1:  # if no targets continue
                continue

            # SGD burn-in
            burnin = min(1000, len(dataloader))
            if (epoch == 1) & (i <= burnin):
                lr = opt.lr * (i / burnin) ** 4
                for g in optimizer.param_groups:
                    g['lr'] = lr

            # Compute loss, compute gradient, update parameters
            loss, components = model(imgs.cuda(), targets.cuda(), targets_len.cuda())
            components = torch.mean(components.view(-1, 5), dim=0)
            loss = torch.mean(loss)
            loss.backward()

            # accumulate gradient for x batches before optimizing
            if ((i + 1) % accumulated_batches == 0) or (i == len(dataloader) - 1):
                optimizer.step()
                optimizer.zero_grad()

            # Running epoch-means of tracked metrics
            ui += 1

            for ii, key in enumerate(model.module.loss_names):
                rloss[key] = (rloss[key] * ui + components[ii]) / (ui + 1)

            # rloss indicates running loss values with mean updated at every epoch
            s = ('%8s%12s' + '%10.3g' * 6) % (
                '%g/%g' % (epoch, sum_epochs),
                '%g/%g' % (i, len(dataloader) - 1),
                rloss['box'], rloss['conf'],
                rloss['id'], rloss['loss'],
                rloss['nT'], time.time() - t0)
            t0 = time.time()
            if i % opt.print_interval == 0:
                logger.info(s)

        # Save latest checkpoint
        checkpoint = {'epoch': epoch,
                      'model': model.module.state_dict(),
                      'optimizer': optimizer.state_dict()}

        # copyfile(cfg, weights_to + '/cfg/yolo3.cfg')
        # copyfile(data_cfg, weights_to + '/cfg/ccmcpe.json')

        latest = osp.join(weights_to, 'latest.pt')
        torch.save(checkpoint, latest)
        """
        if epoch % save_every == 0 and epoch != 0:
            # making the checkpoint lite
            checkpoint["optimizer"] = []
            torch.save(checkpoint, osp.join(weights_to, "weights_epoch_" + str(epoch) + ".pt"))
        """

        # Calculate mAP
        if epoch == 1 or epoch % opt.test_interval == 0:
            with torch.no_grad():
                mAP, R, P = test.test(opt, cfg, data_cfg, weights=latest, batch_size=batch_size, print_interval=40)
                # test.test_emb(cfg, data_cfg, weights=latest, batch_size=batch_size, img_size=img_size,
                              # print_interval=40, nID=dataset.nID)

                if mAP > best_mAP:
                    torch.save(checkpoint, osp.join(weights_to, 'best_mAP.pt'))
                    best_mAP = mAP

                if R > best_Recall:
                    torch.save(checkpoint, osp.join(weights_to, 'best_recall.pt'))
                    best_Recall = R

                final_result.append([epoch, mAP, R, P])
        
        # Call scheduler.step() after opimizer.step() with pytorch > 1.1.0
        scheduler.step()

    print('Epoch,   mAP,   R,   P:')
    for row in final_result:
        print(row[0], row[1][0], row[2][0], row[3][0])

    print(f"best mAP:  {best_mAP}")
    print(f"best Recall:  {best_Recall}")

    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--backbone', type=str, default='swin', help='backbone, darknet or swin transformer')

    parser.add_argument('--epochs', type=int, default=40, help='number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='size of each image batch')
    parser.add_argument('--accumulated-batches', type=int, default=1, help='number of batches before optimizer step')
    parser.add_argument('--cfg', type=str, default='cfg/yolov3_1088x608_newanchor3-fullswin.cfg', help='cfg file path')
    parser.add_argument('--weights-from', type=str, default='weights/',
                        help='Path for getting the trained model for resuming training (Should only be used with '
                             '--resume)')
    parser.add_argument('--weights-to', type=str, default='weights/',
                        help='Store the trained weights after resuming training session. It will create a new folder '
                             'with timestamp in the given path')
    parser.add_argument('--save-model-after', type=int, default=5,
                        help='Save a checkpoint of model at given interval of epochs')
    parser.add_argument('--data-cfg', type=str, default='cfg/ccmcpe.json', help='coco.data file path')
    parser.add_argument('--img-size', type=int, default=[1088, 608], nargs='+', help='pixels')
    parser.add_argument('--resume', action='store_true', help='resume training flag')
    parser.add_argument('--print-interval', type=int, default=40, help='print interval')
    parser.add_argument('--test-interval', type=int, default=3, help='test interval')
    parser.add_argument('--lr', type=float, default=3e-4, help='init lr')
    parser.add_argument('--unfreeze-bn', action='store_true', help='unfreeze bn')
    parser.add_argument('--eval_only', action='store_true', help='for debug')
    opt = parser.parse_args()

    init_seeds()

    train(
        opt.cfg,
        opt.data_cfg,
        weights_from=opt.weights_from,
        weights_to=opt.weights_to,
        save_every=opt.save_model_after,
        img_size=opt.img_size,
        resume=opt.resume,
        epochs=opt.epochs,
        batch_size=opt.batch_size,
        accumulated_batches=opt.accumulated_batches,
        opt=opt,
    )
