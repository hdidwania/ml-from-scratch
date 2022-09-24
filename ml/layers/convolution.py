"""
Implementation of convolution layer
"""

import numpy as np

from ml.layers.base import LearnableLayer


# References :
# https://agustinus.kristia.de/techblog/2016/07/16/convnet-conv-layer/
# https://www.youtube.com/watch?v=Lakz2MoHy6o
# https://datascience-enthusiast.com/DL/Convolution_model_Step_by_Stepv2.html


def zero_pad(x, pad):
    B, H, W, C = x.shape
    padded_output = np.zeros((B, H + 2 * pad, W + 2 * pad, C))
    padded_output[:, pad:-pad, pad:-pad, :] = x
    return padded_output


"""
ISSUES
1- Padding logic is messed up, recheck forward and backward when padding
2- Looping makes it too slow, find better way to implement
"""


class Conv2D(LearnableLayer):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.params["w"] = np.random.randn(
            out_channels, kernel_size, kernel_size, in_channels
        )
        self.params["b"] = np.zeros([1, out_channels])

    def forward(self, x):
        self.buffer["x"] = x
        B, H, W, _ = x.shape
        out_dims = (
            B,
            (H + 2 * self.padding - self.kernel_size) // self.stride + 1,
            (W + 2 * self.padding - self.kernel_size) // self.stride + 1,
            self.out_channels,
        )
        if self.padding:
            x = zero_pad(x, self.padding)
        x_ = self.flatten_img(x)  # (n_positions * B, flat patch length)
        w_ = self.flatten_weights()  # (out_channels, flat patch length)
        x_out = (
            np.dot(x_, w_.transpose()) + self.params["b"]
        )  # (n_positions * B, out_channels)
        x_out = np.reshape(x_out, out_dims)  # Unflatten
        self.buffer["x_"] = x_
        self.buffer["w_"] = w_
        return x_out

    def backward(self, dout):
        dout = np.reshape(
            dout, [-1, self.out_channels]
        )  # Flatten : (n_positions * B, out_channels)
        dx_ = np.dot(dout, self.buffer["w_"])  # (n_positions * B, flat patch length)
        dw_ = np.dot(
            dout.transpose(), self.buffer["x_"]
        )  # (out_channels, flat patch length)
        db = np.sum(dout, axis=0, keepdims=True)
        dw = np.reshape(
            dw_,
            [self.out_channels, self.kernel_size, self.kernel_size, self.in_channels],
        )
        din = np.zeros(self.buffer["x"].shape)
        B, H, W, _ = din.shape
        index = 0
        for i_b in range(B):
            for i_h in range(0, H, self.stride):
                if i_h + self.kernel_size > H:
                    continue
                for i_w in range(0, W, self.stride):
                    if i_w + self.kernel_size > W:
                        continue
                    din[
                        i_b,
                        i_h : i_h + self.kernel_size,
                        i_w : i_w + self.kernel_size,
                        :,
                    ] += np.reshape(
                        dx_[index], [self.kernel_size, self.kernel_size, -1]
                    )
                    index += 1
        self.grad["w"] = dw
        self.grad["b"] = db
        return din

    def flatten_img(self, x):
        B, H, W, _ = x.shape
        flattened_patches = list()
        for i_b in range(B):
            for i_h in range(0, H, self.stride):
                if i_h + self.kernel_size > H:
                    continue
                for i_w in range(0, W, self.stride):
                    if i_w + self.kernel_size > W:
                        continue
                    patch = x[
                        i_b,
                        i_h : i_h + self.kernel_size,
                        i_w : i_w + self.kernel_size,
                        :,
                    ]
                    flattened_patches.append(np.reshape(patch, [-1]))
        flattened_patches = np.stack(flattened_patches, axis=0)
        return flattened_patches

    def flatten_weights(self):
        return np.reshape(self.params["w"], [self.out_channels, -1])
