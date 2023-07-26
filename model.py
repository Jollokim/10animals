import torch
import torch.nn as nn

import torch.functional as F


class Inception(nn.Module):
    def __init__(self, in_channels, n_classes, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.intro = IntroBlock(in_channels, 192)

        self.stage1 = nn.Sequential(
            InceptionModule(192, (64, 128, 32), (96, 16)),
            InceptionModule(224, (128, 192, 96), (128, 32)),
            nn.MaxPool2d(3, 2),
            InceptionModule(416, (192, 208, 48), (96, 16))
        )

        self.aux1 = AuxiliaryHead(n_classes)

        self.stage2 = nn.Sequential(
            InceptionModule(448, (160, 224, 64), (112, 24)),
            InceptionModule(448, (128, 256, 64), (128, 24)),
            InceptionModule(448, (112, 288, 64), (144, 32))
        )

        self.aux2 = AuxiliaryHead(n_classes)

        self.stage3 = nn.Sequential(
            InceptionModule(464, (256, 320, 128), (160, 32)),
            nn.MaxPool2d(3, 2),
            InceptionModule(704, (256, 320, 128), (160, 32)),
            InceptionModule(704, (384, 384, 128), (192, 48))
        )

        self.head = AuxiliaryHead(n_classes)

    def forward(self, x):
        x = self.intro(x)

        # print()
        # print('intro out', x.shape)
        # print()

        x = self.stage1(x)

        # print()
        # print('stage1 out', x.shape)
        # print()

        x_clone = torch.clone(x)
        aux1_out = self.aux1(x_clone)

        x = self.stage2(x)

        # print()
        # print('stage2 out', x.shape)
        # print()

        x_clone = torch.clone(x)
        aux2_out = self.aux2(x_clone)

        x = self.stage3(x)

        # print()
        # print('stage3 out', x.shape)
        # print()

        head_out = self.head(x)

        return aux1_out, aux2_out, head_out


class IntroBlock(nn.Module):
    def __init__(self, in_channels, out_channels, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.intro = nn.Sequential(
            ConvBlock(in_channels, 64, kernel_size=7, stride=2),
            nn.MaxPool2d(3, 2),

            ConvBlock(64, out_channels, kernel_size=3),
            # nn.MaxPool2d(3, 2) just to make the feature map size a bit bigger because of missing branch in InceptionModules.
        )

    def forward(self, x):
        return self.intro(x)


class InceptionModule(nn.Module):
    def __init__(self, in_channels, outchannels: tuple[int, int, int], outchannels_reduce: tuple[int, int], *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.branch1 = nn.Sequential(
            ConvBlock(in_channels, outchannels[0], kernel_size=1)
        )

        self.branch2 = nn.Sequential(
            ConvBlock(
                in_channels, outchannels_reduce[0], kernel_size=1, padding='same'),
            ConvBlock(
                outchannels_reduce[0], outchannels[1], kernel_size=3, padding='same')
        )

        self.branch3 = nn.Sequential(
            ConvBlock(
                in_channels, outchannels_reduce[1], kernel_size=1, padding='same'),
            ConvBlock(
                outchannels_reduce[1], outchannels[2], kernel_size=5, padding='same')
        )

    def forward(self, x):
        branch_x1 = self.branch1(x)
        branch_x2 = self.branch2(x)
        branch_x3 = self.branch3(x)

        out = torch.cat((branch_x1, branch_x2, branch_x3), dim=1)

        return out


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.conv = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)

        return x


class SpatialPyramidPooling(nn.Module):
    def __init__(self, levels, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.levels = levels

    def forward(self, x):
        # print('spatial', x.shape)

        new_x = None
        for level in self.levels:
            kernel_h = x.shape[2] // level
            kernel_w = x.shape[3] // level

            pooled = nn.MaxPool2d((kernel_h, kernel_w),
                                  stride=(kernel_h, kernel_w))(x)
            pooled = pooled.flatten(start_dim=1)

            # print('pooled level', level, pooled.shape)

            if new_x is None:
                new_x = pooled
            else:
                new_x = torch.cat((new_x, pooled), dim=1)

        return new_x


class AuxiliaryHead(nn.Module):
    def __init__(self, n_classes, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.pyramidpooling = SpatialPyramidPooling([1, 2, 3])

        self.head = nn.Sequential(
            nn.LazyLinear(4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.LazyLinear(n_classes),
            nn.Softmax(1)
        )

    def forward(self, x):
        x = self.pyramidpooling(x)

        # print('after pyramidpooling', x.shape)

        x = self.head(x)

        return x


if __name__ == '__main__':
    # model = InceptionModule(192, (64, 128, 32), (96, 16))
    model = Inception(3, 10)

    batch_x = torch.randn((5, 3, 500, 255))

    print(batch_x.shape)

    y = model(batch_x)

    for out in y:
        print(out.shape)
