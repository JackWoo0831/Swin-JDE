"""
打印Swin-T前向传播过程中维度变化
"""

from swin_transformer import * 

if __name__ == '__main__':

    test_tensor = torch.randint(0, 255, size=(1, 3, 224, 224))

    model = SwinTransformer()

    model.forward(test_tensor)

