FAQ
---

Listed below are a number of common issues users face with the various parts of
the C++ API.

C++ Extensions
==============

Undefined symbol errors from PyTorch/ATen
*****************************************

**Problem**: You import your extension and get an ``ImportError`` stating that
some C++ symbol from PyTorch or ATen is undefined. For example::

  >>> import extension
  Traceback (most recent call last):
    File "<stdin>", line 1, in <module>
  ImportError: /home/user/.pyenv/versions/3.7.1/lib/python3.7/site-packages/extension.cpython-37m-x86_64-linux-gnu.so: undefined symbol: _ZN2at19UndefinedTensorImpl10_singletonE

**Fix**: The fix is to ``import torch`` before you import your extension. This will make
the symbols from the PyTorch dynamic (shared) library that your extension
depends on available, allowing them to be resolved once you import your extension.

I created a tensor using a function from ``at::`` and get errors
****************************************************************

**Problem**: You created a tensor using e.g. ``at::ones`` or ``at::randn`` or
any other tensor factory from the ``at::`` namespace and are getting errors.

**Fix**: Replace ``at::`` with ``torch::`` for factory function calls. You
should never use factory functions from the ``at::`` namespace, as they will
create tensors. The corresponding ``torch::`` functions will create variables,
and you should only ever deal with variables in your code.

LibTorch
========

How do I move a model to GPU?
*****************************

**Problem**: You want to run your model on GPU but are unsure how to move both
the model and tensors to the correct device.

**Fix**: Use the ``to()`` method to move your model and tensors to a CUDA device::

  torch::Device device(torch::kCUDA);
  model->to(device);
  auto input = torch::randn({1, 3, 224, 224}).to(device);
  auto output = model->forward(input);

You can also check for CUDA availability before moving::

  torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);


Make sure to compile with the TorchScript headers by including ``<torch/script.h>``.

My model runs slower in C++ than in Python
******************************************

**Problem**: Your model inference is slower in C++ compared to Python.

**Fix**: There are several common causes:

1. **Enable inference mode**: Wrap your inference code with ``torch::NoGradGuard``
   to disable gradient computation::

     torch::NoGradGuard no_grad;
     auto output = model->forward(input);

2. **Enable optimizations**: For TorchScript models, use ``optimize_for_inference``::

     module = torch::jit::optimize_for_inference(module);

3. **Warm up the model**: Run a few inference passes before benchmarking to allow
   JIT compilation and memory allocation to complete.

4. **Check thread settings**: Ensure proper thread configuration::

     at::set_num_threads(4);  // Adjust based on your hardware

Neural Network Modules
======================

How do I register submodules in a custom module?
************************************************

**Problem**: You created a custom module but the submodules are not being
recognized during ``forward()`` or when saving/loading the model.

**Fix**: You must register submodules in the constructor using
``register_module()``::

  struct MyModel : torch::nn::Module {
    MyModel() {
      fc1 = register_module("fc1", torch::nn::Linear(784, 128));
      fc2 = register_module("fc2", torch::nn::Linear(128, 10));
    }

    torch::Tensor forward(torch::Tensor x) {
      x = torch::relu(fc1->forward(x));
      return fc2->forward(x);
    }

    torch::nn::Linear fc1{nullptr}, fc2{nullptr};
  };

How do I set a module to evaluation mode?
*****************************************

**Problem**: Layers like Dropout and BatchNorm behave differently during training
and evaluation, and you need to switch between modes.

**Fix**: Use the ``eval()`` and ``train()`` methods::

  model->eval();  // Set to evaluation mode
  // ... run inference ...
  model->train(); // Set back to training mode

Data Loading
============

How do I create a custom dataset?
*********************************

**Problem**: You want to load your own data instead of using built-in datasets.

**Fix**: Create a class that inherits from ``torch::data::datasets::Dataset`` and
implement the ``get()`` and ``size()`` methods::

  class CustomDataset : public torch::data::datasets::Dataset<CustomDataset> {
   public:
    explicit CustomDataset(const std::string& data_path) {
      // Load your data here
    }

    torch::data::Example<> get(size_t index) override {
      // Return a single data sample
      torch::Tensor data = /* load data at index */;
      torch::Tensor label = /* load label at index */;
      return {data, label};
    }

    torch::optional<size_t> size() const override {
      return dataset_size_;
    }

   private:
    size_t dataset_size_;
  };

Then use it with a DataLoader::

  auto dataset = CustomDataset("path/to/data")
    .map(torch::data::transforms::Stack<>());
  auto dataloader = torch::data::make_data_loader(
    std::move(dataset),
    torch::data::DataLoaderOptions().batch_size(32).workers(4));

Serialization
=============

How do I save and load model weights?
*************************************

**Problem**: You want to save trained model weights and load them later.

**Fix**: Use ``torch::save()`` and ``torch::load()``::

  // Saving
  torch::save(model, "model.pt");

  // Loading
  torch::load(model, "model.pt");

For saving only specific tensors or state::

  torch::serialize::OutputArchive archive;
  model->save(archive);
  archive.save_to("model_weights.pt");

  // Loading
  torch::serialize::InputArchive archive;
  archive.load_from("model_weights.pt");
  model->load(archive);

Build and Compilation
=====================

CMake cannot find Torch
***********************

**Problem**: When building your project with CMake, you get an error that
``Torch`` package cannot be found.

**Fix**: You need to specify the path to the LibTorch installation using
``CMAKE_PREFIX_PATH``::

  cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch ..

Alternatively, set ``Torch_DIR`` to point to the directory containing
``TorchConfig.cmake``::

  cmake -DTorch_DIR=/path/to/libtorch/share/cmake/Torch ..

Linker errors with undefined references
***************************************

**Problem**: Your project compiles but you get linker errors with undefined
references to PyTorch symbols.

**Fix**: Ensure you're linking against all required libraries in your
``CMakeLists.txt``::

  find_package(Torch REQUIRED)
  add_executable(my_app main.cpp)
  target_link_libraries(my_app "${TORCH_LIBRARIES}")
  set_property(TARGET my_app PROPERTY CXX_STANDARD 17)

Also ensure that the compiler flags are set correctly::

  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
