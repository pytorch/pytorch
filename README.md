![PyTorch Logo](https://github.com/pytorch/pytorch/blob/main/docs/source/_static/img/pytorch-logo-dark.png)

--------------------------------------------------------------------------------

PyTorch is a Python package offering two high-level features:
- Tensor computation (similar to NumPy) with strong GPU acceleration
- Deep neural networks constructed on a tape-based autograd system

You can integrate your favorite Python packages such as NumPy, SciPy, and Cython with PyTorch when needed.

Find our Continuous Integration status at [hud.pytorch.org](https://hud.pytorch.org/ci/pytorch/pytorch/main).

<!-- toc -->

- [More About PyTorch](#more-about-pytorch)
  - [A GPU-Ready Tensor Library](#a-gpu-ready-tensor-library)
  - [Dynamic Neural Networks: Tape-Based Autograd](#dynamic-neural-networks-tape-based-autograd)
  - [Python First](#python-first)
  - [Imperative Experiences](#imperative-experiences)
  - [Fast and Lean](#fast-and-lean)
  - [Extensions Without Pain](#extensions-without-pain)
- [Installation](#installation)
  - [Binaries](#binaries)
    - [NVIDIA Jetson Platforms](#nvidia-jetson-platforms)
  - [From Source](#from-source)
    - [Prerequisites](#prerequisites)
    - [Install Dependencies](#install-dependencies)
    - [Get the PyTorch Source](#get-the-pytorch-source)
    - [Install PyTorch](#install-pytorch)
      - [Adjust Build Options (Optional)](#adjust-build-options-optional)
  - [Docker Image](#docker-image)
    - [Using pre-built images](#using-pre-built-images)
    - [Building the image yourself](#building-the-image-yourself)
  - [Building the Documentation](#building-the-documentation)
  - [Previous Versions](#previous-versions)
- [Getting Started](#getting-started)
- [Resources](#resources)
- [Communication](#communication)
- [Releases and Contributing](#releases-and-contributing)
- [The Team](#the-team)
- [License](#license)

<!-- tocstop -->

## More About PyTorch

[Learn the basics of PyTorch](https://pytorch.org/tutorials/beginner/basics/intro.html)

PyTorch is essentially a library comprising the following components:

| Component | Description |
|-----------|-------------|
| [**torch**](https://pytorch.org/docs/stable/torch.html) | A Tensor library akin to NumPy, but with robust GPU support |
| [**torch.autograd**](https://pytorch.org/docs/stable/autograd.html) | A tape-based automatic differentiation library supporting all differentiable Tensor operations in torch |
| [**torch.jit**](https://pytorch.org/docs/stable/jit.html) | A compilation stack (TorchScript) for creating serializable and optimizable models from PyTorch code |
| [**torch.nn**](https://pytorch.org/docs/stable/nn.html) | A neural networks library deeply integrated with autograd, designed for flexibility |
| [**torch.multiprocessing**](https://pytorch.org/docs/stable/multiprocessing.html) | Python's multiprocessing with seamless memory sharing of torch Tensors across processes, useful in data loading and Hogwild training |
| [**torch.utils**](https://pytorch.org/docs/stable/data.html) | DataLoader and other utility functions for convenience |

PyTorch is popularly used either as:
- A substitute for NumPy to leverage the power of GPUs.
- A deep learning research platform that provides maximum flexibility and speed.

Elaborating on the features:

### A GPU-Ready Tensor Library

PyTorch provides Tensors that can reside on the CPU or the GPU, enhancing the computation speed significantly.

We offer a comprehensive range of tensor routines that cater to your scientific computing needs like slicing, indexing, mathematical operations, linear algebra, and reductions. They are optimized for speed!

### Dynamic Neural Networks: Tape-Based Autograd

PyTorch's innovative approach involves using a tape recorder to track operations and then replaying them for backward computation.

Most frameworks, such as TensorFlow, Theano, Caffe, and CNTK, operate with a static view of the world, meaning the network's structure is fixed and unchangeable once defined. PyTorch's dynamic nature allows you to change the network architecture on the fly without starting from scratch.

This feature, while not exclusive to PyTorch, is among the fastest implementations available.

![Dynamic graph](https://github.com/pytorch/pytorch/blob/main/docs/source/_static/img/dynamic_graph.gif)

### Python First

Designed to integrate seamlessly with Python, PyTorch allows you to use it naturally like you would with libraries such as NumPy, SciPy, or scikit-learn. It’s built to be deeply interoperable with Python, enabling you to use familiar syntax and libraries.

### Imperative Experiences

PyTorch is intuitive and straightforward to use, making your workflow much simpler and more linear. The execution is immediate, making debugging easy and the user experience much more gratifying.

### Fast and Lean

Optimized to be both speedy and memory-efficient across both small and large-scale neural networks, PyTorch makes use of its robust, underlying libraries, including Intel MKL and NVIDIA's cuDNN and NCCL, to maximize speed. It’s designed to be lightweight, with minimal framework overhead.

### Extensions Without Pain

PyTorch is designed to provide an easy pathway to create new neural network modules and interfaces with its Tensor API. Extending PyTorch should be effortless, whether you’re writing your modules in Python or utilizing external libraries.

## Installation

[Detailed installation instructions and binaries are available on our website.](https://pytorch.org/get-started/locally/)

## Getting Started

Kickstart your PyTorch learning journey with these resources:
- [Tutorials: Comprehensive guides to get started with PyTorch](https://pytorch.org/tutorials/)
- [Examples: Understandable PyTorch code across various domains](https://github.com/pytorch/examples)
- [API Reference: Detailed descriptions and support for all PyTorch functions](https://pytorch.org/docs/)
- [Glossary: Terms and definitions related to PyTorch](https://github.com/pytorch/pytorch/blob/main/GLOSSARY.md)

## Resources

Explore these resources to learn more about PyTorch and stay connected with the community:
- [PyTorch Website](https://pytorch.org/)
- [Tutorials, Examples, Models, and more](https://pytorch.org/resources)

## Communication

Stay engaged with the PyTorch community:
- [Discuss implementations, research, etc. on the Forums](https://discuss.pytorch.org)
- [Report bugs, request features, and more on GitHub Issues](https://github.com/pytorch/pytorch/issues)

## Releases and Contributing

We welcome all contributions. For bug fixes, please open a pull request. For new features, please discuss with us via GitHub issues before submitting a pull request. Review our [Contribution Guide](CONTRIBUTING.md) for more details.

## The Team

PyTorch is maintained by a group of proficient engineers and researchers, with contributions from a vibrant community.

## License

Refer to the [LICENSE](LICENSE) file for the BSD-style license details.
