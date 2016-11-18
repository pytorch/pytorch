# Module

This is the base class for all Modules defined in the nn package.

```python
# .parameters()
```

```python
>>> for param in model.parameters():
>>>     print(type(param.data), param.size())
<class 'torch.FloatTensor'> (20L,)
<class 'torch.FloatTensor'> (20L, 1L, 5L, 5L)
```

```python
```

```python
# .parameter_dict()
```

```python
>>> pdict = model.parameter_dict()
>>> print(pdict.keys())
['bias', 'weight']
```

```python
```

Even the Container class derives from this class.

An nn.Module has the following interface:

**Constructor:**
   nn.Module(**parameters)

All arguments passed in to the constructor need to be of type 
nn.Parameter or a Tensor.


**forward(...)**

This is the function that one defines when subclassing to create
their own modules.
It takes in inputs and returns outputs.

**__call__(...)**

This calls the forward function, as well as the hooks

**register_buffer(name, tensor)**

This is typically used to register a buffer that is not a Parameter.
For example, in BatchNorm, the running_mean is a buffer, so one would
register it in the constructor of BatchNorm with:

`self.register_buffer('running_mean', torch.zeros(num_features))`

The registered buffers can simply be accessed as class members
when needed.

**cpu()**

Recursively moves all it's parameters and buffers to the CPU

**cuda(device_id=None)**
Recursively moves all it's parameters and buffers to the CUDA memory.
If device_id is given, moves it to GPU number device_id

**float()**
Typecasts the parameters and buffers to float

**double()**
Typecasts the parameters and buffers to double

**register_forward_hook(name, hook)**

This will register a user-defined closure on the module.
Whenever the module finishes it's forward operation,
the user closure is called.
The signature of the closure is `def closure(input, output)`

**register_backward_hook(name, hook)**

This will register a user-defined closure on the module.
Whenever the module finishes it's backward operation,
the user closure is called.
The signature of the closure is `def closure(gradOutput, gradInput)`

**remove_forward_hook(name)**

Removes a registered forward hook with the given name

**remove_backward_hook(name)**

Removes a registered backward hook with the given name

**`[generator] parameters()`**

returns a generator over all learnable parameters in the container instance. 
This can typically be passed to the optimizer API

**`[dict] parameter_dict()`**

returns a dictionary of learnable parameters of the Module.
For example: ['weight' : Parameter(torch.FloatTensor(20x1x5x5)),
              'bias'   : Parameter(torch.FloatTensor(20)),
             ]

**`load_parameter_dict(dict)`**

Given a parameter dict, sets the parameters of self to be the given dict.

**`train()`**

Sets the Container to training mode (for modules such as batchnorm, dropout etc.)

**`eval()`**

Sets the Container to evaluate mode (for modules such as batchnorm, dropout etc.)

**`zero_grad()`**

Zeroes the gradients of each Parameter of the module
# Container

This is the base container class for all neural networks you would define.

```python
# Example of using Container
 class Net(nn.Container):
    def __init__(self):
        super(Net, self).__init__(
            conv1 = nn.Conv2d(1, 20, 5),
            relu  = nn.ReLU()
         )
    def forward(self, input):
        output = self.relu(self.conv1(x))
        return output
 model = Net()
```

```python
# one can add modules to the container after construction
model.add_module('pool1', nn.MaxPool2d(2, 2))
```

```python
```

```python
# .parameters()
```

```python
>>> for param in model.parameters():
>>>     print(type(param.data), param.size())
<class 'torch.FloatTensor'> (20L,)
<class 'torch.FloatTensor'> (20L, 1L, 5L, 5L)
```

```python
```

```python
# .parameter_dict()
```

```python
>>> pdict = model.parameter_dict()
>>> print(pdict.keys())
['conv1.bias', 'conv1.weight']
```

```python
```

You will subclass your container from this class.
In the constructor you define the modules that you would want to use,
and in the "forward" function you use the constructed modules in
your operations.

To make it easier to understand, given is a small example.

One can also add new modules to a container after construction.
You can do this with the add_module function 
or by assigning them as Container attributes.

## one can also set modules as attributes of the container
model.conv1 = nn.Conv2d(12, 24, 3)
The container has some important additional methods: 

**`[generator] parameters()`**

returns a generator over all learnable parameters in the container instance. 
This can typically be passed to the optimizer API

**`[dict] parameter_dict()`**

returns a dictionary of learnable parameters of the Container.
For example: ['conv1.weight' : Parameter(torch.FloatTensor(20x1x5x5)),
              'conv1.bias'   : Parameter(torch.FloatTensor(20)),
             ]


**`load_parameter_dict(dict)`**

Given a parameter dict, sets the parameters of self to be the given dict.
It loads loads the parameters recursively.
Excessive or non-matching parameter names are ignored.
For example, the input dict has an entry 'conv44.weight', but 
if the container does not have a module named 'conv44', then this entry is ignored.

**`children()`**

Returns a generator over all the children modules of self

**`train()`**

Sets the Container (and all it's child modules) to training mode (for modules such as batchnorm, dropout etc.)

**`eval()`**

Sets the Container (and all it's child modules) to evaluate mode (for modules such as batchnorm, dropout etc.)

**`apply(closure)`**

Applies the given closure to each parameter of the container. 


**__Note: Apart from these, the container will define the base functions that it has derived from nn.Module __**
