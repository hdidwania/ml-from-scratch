# ML From Scratch
This repo contains my implementations of machine learning algorithms from scratch using numpy.

The idea behind this project is to gain a deeper understanding to different ML algorithms, tricks, and practices, by implementing them from scratch and seeing their impacts. The project tries to approach ML from a very first-principles view, trying to uncover the core ideas hidden behind the abstractions provided by many libraries.

## Gradient Checking
Every layer has a logic for forward pass and backward pass. The best way to check the correctness of the implementation is to do a gradient check. The gradients for a layer can be checked using following command:
```
python grad_check_<>.py --function <function name>
```

## Implementations
The following mini projects are implemented:
1. Classification on MNIST data using a dense network.


## Datasets Used
You can find the datasets used in the scripts from the following links:
- [MNIST](https://pjreddie.com/projects/mnist-in-csv/)
