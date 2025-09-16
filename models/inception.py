import torch
import torch.nn as nn


class InceptionBlock(nn.Module):
    def __init__(self, in_channels, n_filters, kernel_sizes=[3, 5, 7], bottleneck_channels=32):
        super(InceptionBlock, self).__init__()

        self.bottleneck = nn.Conv2d(
            in_channels, bottleneck_channels, kernel_size=1, bias=False)

        self.convs = nn.ModuleList()
        for kernel_size in kernel_sizes:
            padding = kernel_size // 2
            self.convs.append(nn.Conv2d(
                bottleneck_channels, n_filters,
                kernel_size=kernel_size, padding=padding))

        self.maxpool = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, n_filters, kernel_size=1)
        )

        self.bn = nn.BatchNorm2d(n_filters * (len(kernel_sizes) + 1))
        self.relu = nn.ReLU()

    def forward(self, x):
        x_bottleneck = self.bottleneck(x)

        conv_outputs = []
        for conv in self.convs:
            conv_outputs.append(conv(x_bottleneck))

        maxpool_output = self.maxpool(x)

        combined = torch.cat(conv_outputs + [maxpool_output], dim=1)
        output = self.bn(combined)
        output = self.relu(output)
        return output
