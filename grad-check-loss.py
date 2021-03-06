"""
Gradient check to verify backprop of loss functions
"""
import argparse

import numpy as np
from numpy.linalg import norm

from ml.loss import BCELoss, MSELoss

EPS = 1e-5


def check_mse():
    dim = int(input("Enter vector dimensions: "))
    batch_size = int(input("Enter batch size: "))

    true = np.random.randn(batch_size, dim)
    pred = np.random.randn(batch_size, dim)

    mse = MSELoss()
    _ = mse.forward(true, pred)

    dpred = mse.backward()

    dpred_man = np.zeros([batch_size, dim])
    for b in range(batch_size):
        for i in range(dim):
            h = np.zeros([batch_size, dim])
            h[b, i] = EPS
            dpred_man[b, i] = (
                mse.forward(true, pred + h) - mse.forward(true, pred - h)
            ) / (2 * EPS)

    diff = dpred_man - dpred
    print("Norm of difference:")
    print(norm(diff))


def check_bce():
    dim = int(input("Enter vector dimensions: "))
    batch_size = int(input("Enter batch size: "))

    true = np.random.choice([0, 1], (batch_size, dim))
    pred = np.random.uniform(0, 1, (batch_size, dim))

    bce = BCELoss()
    _ = bce.forward(true, pred)

    dpred = bce.backward()

    dpred_man = np.zeros([batch_size, dim])
    for b in range(batch_size):
        for i in range(dim):
            h = np.zeros([batch_size, dim])
            h[b, i] = EPS
            dpred_man[b, i] = (
                bce.forward(true, pred + h) - bce.forward(true, pred - h)
            ) / (2 * EPS)

    diff = dpred_man - dpred
    print("Norm of difference:")
    print(norm(diff))


args_to_fn = {"mse": check_mse, "bce": check_bce}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--function", required=True)
    args = parser.parse_args()

    check_fn = args_to_fn[args.function]
    check_fn()
