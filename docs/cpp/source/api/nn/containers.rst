Containers
==========

Container modules hold other modules and define how they are composed together.
Use containers to build complex architectures from simpler building blocks.

- **Sequential**: Chain modules in order, output of one feeds into the next
- **ModuleList**: Store modules in a list for iteration (not auto-forwarded)
- **ModuleDict**: Store modules in a dictionary for named access
- **ParameterList/ParameterDict**: Store parameters directly without wrapping in modules

.. note::
   PyTorch's C++ API uses the PIMPL (Pointer to Implementation) pattern. You create
   modules using the public class name (e.g., ``torch::nn::Sequential``), which
   internally wraps an implementation class (``SequentialImpl``). The documentation
   below shows the implementation classes, which contain all the actual methods.

Sequential
----------

``Sequential`` is a container that chains modules together. Each module's output
becomes the next module's input. This is the simplest way to build feed-forward
networks.

.. doxygenclass:: torch::nn::Sequential
   :members:
   :undoc-members:

.. doxygenclass:: torch::nn::SequentialImpl
   :members:
   :undoc-members:

**Example:**

.. code-block:: cpp

   torch::nn::Sequential seq(
       torch::nn::Conv2d(torch::nn::Conv2dOptions(1, 32, 3)),
       torch::nn::ReLU(),
       torch::nn::Conv2d(torch::nn::Conv2dOptions(32, 64, 3)),
       torch::nn::ReLU()
   );

   auto output = seq->forward(input);

ModuleList
----------

``ModuleList`` stores modules in a list for indexed or iterated access. Unlike
``Sequential``, it does not have a built-in ``forward()`` method—you control how
modules are called.

.. doxygenclass:: torch::nn::ModuleList
   :members:
   :undoc-members:

.. doxygenclass:: torch::nn::ModuleListImpl
   :members:
   :undoc-members:

**Example:**

.. code-block:: cpp

   torch::nn::ModuleList layers;
   layers->push_back(torch::nn::Linear(10, 20));
   layers->push_back(torch::nn::Linear(20, 30));

   torch::Tensor x = input;
   for (const auto& layer : *layers) {
       x = layer->as<torch::nn::Linear>()->forward(x);
   }

ModuleDict
----------

``ModuleDict`` stores modules in a dictionary for named access. Useful when you
need to select modules by name at runtime.

.. doxygenclass:: torch::nn::ModuleDict
   :members:
   :undoc-members:

.. doxygenclass:: torch::nn::ModuleDictImpl
   :members:
   :undoc-members:

ParameterList
-------------

``ParameterList`` stores parameters directly without wrapping them in modules.

.. doxygenclass:: torch::nn::ParameterList
   :members:
   :undoc-members:

.. doxygenclass:: torch::nn::ParameterListImpl
   :members:
   :undoc-members:

ParameterDict
-------------

``ParameterDict`` stores parameters in a dictionary for named access.

.. doxygenclass:: torch::nn::ParameterDict
   :members:
   :undoc-members:

.. doxygenclass:: torch::nn::ParameterDictImpl
   :members:
   :undoc-members:
