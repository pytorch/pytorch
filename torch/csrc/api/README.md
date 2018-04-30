# AUTOGRADPP

This is an experimental C++ frontend to pytorch's C++ backend. Use at your own
risk.

How to build:
```
git submodule update --init --recursive

cd pytorch
# On Linux:
python setup.py build
# On macOS (may need to prefix with `MACOSX_DEPLOYMENT_TARGET=10.9 CC=clang CXX=clang++` when using anaconda)
LDSHARED="cc -dynamiclib -undefined dynamic_lookup" python setup.py build

cd ..; mkdir -p build; cd build
cmake .. -DPYTHON_EXECUTABLE:FILEPATH=$(which python)  # helpful if you use anaconda
make -j
```

# Stuff

- Check out the [MNIST example](https://github.com/ebetica/autogradpp/blob/eee977ddd377c484af5fce09ae8676410bb6fcce/tests/integration_t.cpp#L320-L355),
which tries to replicate PyTorch's MNIST model + training loop
- The principled way to write a model is probably something like 
```
AUTOGRAD_CONTAINER_CLASS(MyModel) {
  // This does a 2D convolution, followed by global sum pooling, followed by a linear.
 public:
  void initialize_containers() override {
    myConv_ = add(Conv2d(1, 50, 3, 3).stride(2).make(), "conv");
    myLinear_ = add(Linear(50, 1).make(), "linear");
  }
  variable_list forward(variable_list x) override {
    auto v = myConv_->forward(x);
    v = v.mean(-1).mean(-1);
    return myLinear_.forward({v});
  }
 private:
  Container myLinear_;
  Container myConv_;
}
```

Some things are not implemented:
- SGD, Adagrad, RMSprop, and Adam are the only optimizers implemented
- Bidirectional, batch first, and PackedSequence are not implemented for LSTMs
- Sparse Tensors might work but are very untested

Otherwise, lots of other things work. There may be breaking API changes.
