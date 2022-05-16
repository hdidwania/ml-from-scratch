import argparse

import numpy as np

from ml.layers.loss import MSELoss

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
    print("Difference between backprop gradients and calculated gradients")
    print(diff)
    print("Maximum and minimum differences:")
    print(np.min(diff), np.max(diff))


args_to_fn = {"mse": check_mse}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--function", required=True)
    args = parser.parse_args()

    check_fn = args_to_fn[args.function]
    check_fn()
