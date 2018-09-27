The PyTorch C++ Frontend
========================

The PyTorch C++ frontend is a pure C++11 library for CPU and GPU
tensor computation, with automatic differentation and high level building
blocks for state of the art machine learning applications.

Description
-----------

The PyTorch C++ frontend can be thought of as a C++ version of the
PyTorch Python frontend, providing automatic differentiation and various higher
level abstractions for machine learning and neural networks.  Specifically,
it consists of the following components:

+----------------------+------------------------------------------------------------------------+
| Component            | Description                                                            |
+======================+========================================================================+
| ``torch::Tensor``    | Automatically differentiable, efficient CPU and GPU enabled tensors    |
+----------------------+------------------------------------------------------------------------+
| ``torch::nn``        | A collection of composable modules for neural network modeling         |
+----------------------+------------------------------------------------------------------------+
| ``torch::optim``     | Optimization algorithms like SGD, Adam or RMSprop to train your models |
+----------------------+------------------------------------------------------------------------+
| ``torch::data``      | Datasets, data pipelines and multi-threaded, asynchronous data loader  |
+----------------------+------------------------------------------------------------------------+
| ``torch::serialize`` | A serialization API for storing and loading model checkpoints          |
+----------------------+------------------------------------------------------------------------+
| ``torch::python``    | Glue to bind your C++ models into Python                               |
+----------------------+------------------------------------------------------------------------+
| ``torch::jit``       | Pure C++ access to the TorchScript JIT compiler                        |
+----------------------+------------------------------------------------------------------------+

End-to-end example
------------------

Here is a simple, end-to-end example of defining and training a simple
neural network on the MNIST dataset:

.. code-block:: cpp

  #include <torch/torch.h>

  // Define a new Module.
  struct Net : torch::nn::Module {
    Net() {
      // Construct and register two Linear submodules.
      fc1 = register_module("fc1", torch::nn::Linear(8, 64));
      fc2 = register_module("fc2", torch::nn::Linear(64, 1));
    }

    // Implement the Net's algorithm.
    torch::Tensor forward(torch::Tensor x) {
      // Use one of many tensor manipulation functions.
      x = torch::relu(fc1->forward(x));
      x = torch::dropout(x, /*p=*/0.5);
      x = torch::sigmoid(fc2->forward(x));
      return x;
    }

    // Use one of many "standard library" modules.
    torch::nn::Linear fc1{nullptr}, fc2{nullptr};
  };

  // Create a new Net.
  Net net;

  // Create a multi-threaded data loader for the MNIST dataset.
  auto data_loader =
    torch::data::data_loader(torch::data::datasets::MNIST("./data"));

  // Instantiate an SGD optimization algorithm to update our Net's parameters.
  torch::optim::SGD optimizer(net.parameters(), /*lr=*/0.1);

  for (size_t epoch = 1; epoch <= 10; ++epoch) {
    size_t batch_index = 0;
    // Iterate the data loader to yield batches from the dataset.
    for (auto batch : data_loader) {
      // Reset gradients.
      optimizer.zero_grad();
      // Execute the model on the input data.
      auto prediction = model.forward(batch.data);
      // Compute a loss value to judge the prediction of our model.
      auto loss = torch::binary_cross_entropy(prediction, batch.label);
      // Compute gradients of the loss w.r.t. the parameters of our model.
      loss.backward();
      // Update the parameters based on the calculated gradients.
      optimizer.step();

      if (batch_index++ % 10 == 0) {
        std::cout << "Epoch: " << epoch << " | Batch: " << batch_index
                  << " | Loss: " << loss << std::endl;
        // Serialize your model periodically as a checkpoint.
        torch::save(net, "net.pt");
      }
    }

To see more complete examples of using the PyTorch C++ frontend, see `the example repository
<https://github.com/goldsborough/examples/tree/cpp/cpp>`_.

Philosophy
----------

The PyTorch C++ frontend was originally born out of the necessity to train
machine learning models in low latency, high performance, **pure C++**
environments such as video game engines or production servers. While the Python
frontend to PyTorch provided a flexible, friendly, comparatively fast and
overall formidable solution to machine learning research, its exposure through
Python barred it from reaching the latency guarantees offered by a native, bare
metal language like C++. At the same time, while there were a number of pure
C++ frameworks for machine learning, none were nearly as flexible nor powerful
as PyTorch. As such, the originators of the PyTorch C++ frontend identified a
gap between the superior research experience provided by PyTorch, and the world
of low latencies provided by C++, and thus undertook the creation of a new,
pure C++ frontend to the existing C++ implementations underpinning PyTorch's
Python interface (the "backend"). This frontend is what is presented in this
document. We claim that it succeeds in filling the aforementioned gap, allowing
high performance yet flexible definition and execution of machine learning
models in pure C++, with no Python in the loop what-so-ever.

Owing to its heritage and original goals, the PyTorch C++ frontend is intended
to closely model the Python frontend in its design, naming, conventions and
functionality. We claim that it largely follows through on those intentions.
While there are certainly occasional differences to be found between the two
interfaces, such as cases where we opt not to bring deprecated features or
functions (no matter how popular) into this new, fresh API, we can guarantee to
a high degree that the effort in porting a Python model to C++ lies almost
exclusively in **translating language features**, but **not modifying
functionality or behavior**.

As a corollary to the above, we would also like to note that in many cases
where we were faced between choosing between flexibility and friendliness
towards research versus micro-optimization and robustness in the face of all
possible edge cases, we opted for the former, friendlier path. Flexibility and
dynamism is at the heart of PyTorch, and we aim to preserve this across the
language boundary the C++ frontend bridges.

Lastly, a word of warning: the Python interface is not necessarily slower than
C++. It already calls into C++ for almost anything computationally expensive
(especially any kind of numeric operation). Translating your model to C++ will
usually not make it magically faster, and you may often not notice any
significant performance gain at all! The problem the C++ frontend solves is not
a performance problem. The problem it solves is being able to write friendly,
flexible and intuitive machine learning applications in environments where C++
is your language of choice, or necessity. If you would prefer to write Python,
and can afford to write Python, we recommend using the Python interface to
PyTorch. However, if you would prefer to write C++, or need to write C++, the
C++ frontend to PyTorch provides an API that is approximately as convenient,
flexible, friendly and intuitive as its Python counterpart. The two frontends
serve different use cases, work hand in hand, and neither is meant to
unconditionally replace the other.

Installation
------------

Instructions on how to install the C++ frontend library distribution, including
an example for how to build a minimal application depending on LibTorch, may be
found by following `this <https://pytorch.org/cppdocs/installation.html>`_ link.
