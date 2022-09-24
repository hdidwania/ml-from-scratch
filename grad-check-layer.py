"""
Gradient check to verify backprop of loss functions
"""
import argparse

import numpy as np
from numpy.linalg import norm

from ml.layers.linear import Linear
from ml.layers.convolution import Conv2D
from ml.loss import MSELoss

EPS = 1e-5


def check_linear():
    in_dim = int(input("Enter input dimensions: "))
    out_dim = int(input("Enter output dimensions: "))
    batch_size = int(input("Enter batch size: "))

    x = np.random.randn(batch_size, in_dim)
    y = np.random.randn(batch_size, out_dim)

    layer = Linear(in_dim, out_dim)
    loss_fn = MSELoss()

    def forward(x, y):
        y_pred = layer.forward(x)
        loss = loss_fn.forward(y, y_pred)
        return loss

    def backward():
        dy = loss_fn.backward()
        dx = layer.backward(dy)
        return dx

    _ = forward(x, y)
    din = backward()
    dw = layer.grad["w"]
    db = layer.grad["b"]

    din_man = np.zeros([batch_size, in_dim])
    for b in range(batch_size):
        for i in range(in_dim):
            h = np.zeros([batch_size, in_dim])
            h[b, i] = EPS
            din_man[b, i] = (forward(x + h, y) - forward(x - h, y)) / (2 * EPS)
    diff = din_man - din
    print("Norm of difference for input:")
    print(norm(diff))

    dw_man = np.zeros((in_dim, out_dim))
    for idx_in in range(in_dim):
        for idx_out in range(out_dim):
            h = np.zeros((in_dim, out_dim))
            h[idx_in, idx_out] = EPS
            layer.params["w"] += h
            delta_plus = forward(x, y)
            layer.params["w"] -= 2 * h
            delta_minus = forward(x, y)
            layer.params["w"] += h
            dw_man[idx_in, idx_out] = (delta_plus - delta_minus) / (2 * EPS)
    diff = dw_man - dw
    print("Norm of difference for w:")
    print(norm(diff))

    db_man = np.zeros((1, out_dim))
    for idx_out in range(out_dim):
        h = np.zeros((1, out_dim))
        h[0, idx_out] = EPS
        layer.params["b"] += h
        delta_plus = forward(x, y)
        layer.params["b"] -= 2 * h
        delta_minus = forward(x, y)
        layer.params["b"] += h
        db_man[0, idx_out] = (delta_plus - delta_minus) / (2 * EPS)
    diff = db_man - db
    print("Norm of difference for b:")
    print(norm(diff))


def check_conv2d():
    in_img_dim = int(input("Enter image dimension: "))
    in_channels = int(input("Enter input channels: "))
    out_channels = int(input("Enter output channels: "))
    batch_size = int(input("Enter batch size: "))
    kernel_size = int(input("Enter kernel size: "))
    stride = int(input("Enter stride: "))
    padding = int(input("Enter padding: "))

    out_img_dim = (in_img_dim + 2 * padding - kernel_size) // stride + 1

    x = np.random.randn(batch_size, in_img_dim, in_img_dim, in_channels)
    y = np.random.randn(batch_size, out_img_dim, out_img_dim, out_channels)

    layer = Conv2D(in_channels, out_channels, kernel_size, stride, padding)
    loss_fn = MSELoss()

    def forward(x, y):
        y_pred = layer.forward(x)
        loss = loss_fn.forward(y, y_pred)
        return loss

    def backward():
        dy = loss_fn.backward()
        dx = layer.backward(dy)
        return dx

    _ = forward(x, y)
    din = backward()
    dw = layer.grad["w"]
    db = layer.grad["b"]

    din_man = np.zeros(din.shape)
    for b in range(batch_size):
        for i in range(in_img_dim):
            for j in range(in_img_dim):
                for k in range(in_channels):
                    h = np.zeros(din.shape)
                    h[b, i, j, k] = EPS
                    din_man[b, i, j, k] = (forward(x + h, y) - forward(x - h, y)) / (
                        2 * EPS
                    )
    diff = din_man - din
    print("Norm of difference for input:")
    print(norm(diff))

    dw_man = np.zeros(dw.shape)
    for idx_cout in range(out_channels):
        for idx_kx in range(kernel_size):
            for idx_ky in range(kernel_size):
                for idx_cin in range(in_channels):
                    h = np.zeros(dw.shape)
                    h[idx_cout, idx_kx, idx_ky, idx_cin] = EPS
                    layer.params["w"] += h
                    delta_plus = forward(x, y)
                    layer.params["w"] -= 2 * h
                    delta_minus = forward(x, y)
                    layer.params["w"] += h
                    dw_man[idx_cout, idx_kx, idx_ky, idx_cin] = (
                        delta_plus - delta_minus
                    ) / (2 * EPS)
    diff = dw_man - dw
    print("Norm of difference for w:")
    print(norm(diff))

    db_man = np.zeros(db.shape)
    for idx_out in range(out_channels):
        h = np.zeros(db.shape)
        h[0, idx_out] = EPS
        layer.params["b"] += h
        delta_plus = forward(x, y)
        layer.params["b"] -= 2 * h
        delta_minus = forward(x, y)
        layer.params["b"] += h
        db_man[0, idx_out] = (delta_plus - delta_minus) / (2 * EPS)
    diff = db_man - db
    print("Norm of difference for b:")
    print(norm(diff))


args_to_fn = {"linear": check_linear, "conv2d": check_conv2d}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--function", required=True)
    args = parser.parse_args()

    check_fn = args_to_fn[args.function]
    check_fn()
