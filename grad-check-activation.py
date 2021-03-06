"""
Gradient check to verify backprop of activation functions
"""
import argparse

import numpy as np
from numpy.linalg import norm

from ml.layers.activations import ReLU, Sigmoid

EPS = 1e-5


def check_sigmoid():
    dim = int(input("Enter vector dimensions: "))
    batch_size = int(input("Enter batch size: "))

    x = np.random.randn(batch_size, dim)
    sigmoid = Sigmoid()
    _ = sigmoid.forward(x)

    dy = np.ones((batch_size, dim))
    dx = sigmoid.backward(dy)

    dx_man = np.zeros([batch_size, dim])
    for b in range(batch_size):
        for i in range(dim):
            h = np.zeros([batch_size, dim])
            h[b, i] = EPS
            dx_man[b, i] = (
                (sigmoid.forward(x + h) - sigmoid.forward(x - h)) / (2 * EPS)
            )[b, i]

    diff = dx_man - dx
    print("Norm of difference:")
    print(norm(diff))


def check_relu():
    dim = int(input("Enter vector dimensions: "))
    batch_size = int(input("Enter batch size: "))

    x = np.random.randn(batch_size, dim)
    sigmoid = ReLU()
    _ = sigmoid.forward(x)

    dy = np.ones((batch_size, dim))
    dx = sigmoid.backward(dy)

    dx_man = np.zeros([batch_size, dim])
    for b in range(batch_size):
        for i in range(dim):
            h = np.zeros([batch_size, dim])
            h[b, i] = EPS
            dx_man[b, i] = (
                (sigmoid.forward(x + h) - sigmoid.forward(x - h)) / (2 * EPS)
            )[b, i]

    diff = dx_man - dx
    print("Norm of difference:")
    print(norm(diff))


args_to_fn = {"sigmoid": check_sigmoid, "relu": check_relu}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--function", required=True)
    args = parser.parse_args()

    check_fn = args_to_fn[args.function]
    check_fn()
