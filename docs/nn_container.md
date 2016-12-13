## Containers
### Container

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
# .state_dict()
```

```python
>>> pdict = model.state_dict()
>>> print(sdict.keys())
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

#### one can also set modules as attributes of the container
model.conv1 = nn.Conv2d(12, 24, 3)
The container has some important additional methods: 

**`[generator] parameters()`**

returns a generator over all learnable parameters in the container instance. 
This can typically be passed to the optimizer API

**`[dict] state_dict()`**

returns a dictionary of learnable parameters of the Container.
For example: ['conv1.weight' : Parameter(torch.FloatTensor(20x1x5x5)),
              'conv1.bias'   : Parameter(torch.FloatTensor(20)),
             ]


**`load_state_dict(dict)`**

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
### Sequential

A sequential Container. It is derived from the base nn.Container class

```python
# Example of using Sequential
model = nn.Sequential(
          nn.Conv2d(1,20,5),
          nn.ReLU(),
          nn.Conv2d(20,64,5),
          nn.ReLU()
        )
```

```python
```

Modules will be added to it in the order they are passed in the constructor.
Alternatively, an ordered dict of modules can also be passed in.

To make it easier to understand, given is a small example.
#### Example of using Sequential with OrderedDict
model = nn.Sequential(OrderedDict([
          ('conv1', nn.Conv2d(1,20,5)),
          ('relu1', nn.ReLU()),
          ('conv2', nn.Conv2d(20,64,5)),
          ('relu2', nn.ReLU())
        ]))
