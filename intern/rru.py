"""RRU model implementation in MindSpore."""

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor
import numpy as np


class BasicConv2d(nn.Cell):
    """Basic convolution block with batch normalization and ReLU activation."""

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, has_bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)
        self.relu = nn.ReLU()

    def construct(self, x):
        """Forward pass of the convolution block."""
        x = self.conv(x)
        x = self.bn(x)
        return self.relu(x)


class BasicConv2dLeaky(nn.Cell):
    """Basic convolution block with batch normalization and LeakyReLU activation."""

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2dLeaky, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, has_bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)
        self.leaky_relu = nn.LeakyReLU(alpha=0.1)

    def construct(self, x):
        """Forward pass of the convolution block."""
        x = self.conv(x)
        x = self.bn(x)
        return self.leaky_relu(x)


class RRU(nn.Cell):
    """Recurrent Residual Unit for video processing."""

    def __init__(self, input_size):
        super().__init__()
        self.hidden_size = 256

        bottleneck_size = [256, 128]
        self.reduce_dim_z = BasicConv2d(
            input_size * 2,
            bottleneck_size[0],
            kernel_size=1,
            pad_mode='valid'
        )
        self.s_atten_z = nn.SequentialCell([
            nn.Conv2d(1, bottleneck_size[1], kernel_size=3,
                     pad_mode='same', has_bias=False),
            nn.ReLU(),
            nn.Conv2d(bottleneck_size[1], 64, kernel_size=1,
                     pad_mode='valid', has_bias=False)
        ])
        self.c_atten_z = nn.SequentialCell([
            nn.AdaptiveAvgPool2d((32, 32)),
            nn.Conv2d(bottleneck_size[0], input_size,
                     kernel_size=1, pad_mode='valid', has_bias=False)
        ])
        self.sigmoid = nn.Sigmoid()
        self.mean = ops.ReduceMean(keep_dims=True)
        self.concat = ops.Concat(axis=1)
        self.stack = ops.Stack(axis=2)

    def generate_attention_z(self, x):
        """Generate attention maps for the input feature."""
        z = self.reduce_dim_z(x)
        atten_s = self.s_atten_z(self.mean(z, 1))
        atten_c = self.c_atten_z(z)
        z = self.sigmoid(atten_s * atten_c)
        return z, 1 - z

    def construct(self, x):
        """Forward pass of the RRU model."""
        if len(x.shape) == 4:
            x = ops.ExpandDims()(x, 0)

        depth = x.shape[1]

        first_frame = ops.ExpandDims()(x[:, 0], 1)
        res = self.concat((first_frame, x))
        res = res[:, :-1]
        res = x - res

        h = x[:, 0]
        output = []
        for t in range(depth):
            con_fea = self.concat((h - x[:, t], res[:, t]))
            z_p, z_r = self.generate_attention_z(con_fea)
            h = z_r * h + z_p * x[:, t]
            output.append(h)

        fea = self.stack(output)
        return fea


def main():
    """Main function to demonstrate the RRU model."""
    # Set up the execution environment
    ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")

    # Create random input data
    batch_size = 2
    time_steps = 4
    channels = 64
    height = 32
    width = 32

    # Create random input tensor
    input_data = np.random.rand(
        batch_size, time_steps, channels, height, width
    ).astype(np.float32)
    input_tensor = Tensor(input_data)

    # Initialize the model
    model = RRU(input_size=channels)

    # Forward pass
    output = model(input_tensor)

    print("输入形状:", input_tensor.shape)
    print("输出形状:", output.shape)
    print("模型运行成功!")


if __name__ == '__main__':
    main()



