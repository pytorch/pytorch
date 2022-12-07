# Structured Pruning

## Intro / Motivation

**pruning** is the process of removing weights / connections in a neural network in order to make it run faster.

There are two types of pruning that we support: **structured pruning** and **unstructured pruing**

* **Unstructured pruning** refers to pruning any arbritary weight
* **Structured pruning** refers to pruning by removing entire rows / columns of a weight.

Structured pruning offers several advantages over unstructured pruning at the cost of lower granuality:
1. We can store the masks more efficeintly in structured pruning.
2. Structured pruning relies on removing columns / rows to a speedup whereas unstructured pruning requires custom kernels.

Pruning is usually followed by a retraining or fine-tuning step.

## What do I need to use Structured Pruning?

**Your model must be FX symbolically traceable**

We support structured pruning for the following patterns out of the box.
- linear -> linear
- linear -> activation -> linear
- conv2d -> conv2d
- conv2d -> activation -> conv2d
- conv2d -> activation -> pool -> conv2d
- conv2d -> pool -> activation -> conv2d
- conv2d -> adaptive pool -> flatten -> linear

If you are looking to prune a different pattern, for example `(conv2d -> activation -> pool -> dropout -> conv2d)`, you can do this by creating modifying the pattern dict that we provide to the pruner, see [here](#writing-custom-patterns-and-pruning-functions-for-structured-pruning)

It is up to the user to define the following two items.
1. What layers are you looking to prune
2. Your pruning criteria - basically how you pick weights to prune and to what degree.

## Structured Pruning for Supported Modules
Here is an example script that will prune away 50% of the rows for all the linear layers in the model, based on the saliency of each row.
```python
from torch.ao.pruning._experimental.pruner import BaseStructuredSparsifier
from torch.ao.pruning._experimental.base_structured_sparsifier import _get_default_structured_pruning_patterns
#define model
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(700, 500, bias=True), # Alpha
            nn.ReLU(),
            nn.Linear(500, 800, bias=False), # Beta
            nn.ReLU(),
            nn.Linear(800, 600, bias=True), # Gamma
            nn.ReLU(),
        )
        self.linear = nn.Linear(600, 4, bias=False) # Delta

    def forward(self, x):
        x = self.seq(x)
        x = self.linear(x)
        return x

# define config, specify fqn of layers you want to prune and target sparsity level
pruning_config = [
    {"tensor_fqn": "seq.0.weight", "sparsity_level": 0.5},
    {"tensor_fqn": "seq.2.weight", "sparsity_level": 0.5},
    {"tensor_fqn": "seq.4.weight", "sparsity_level": 0.5},
    {"tensor_fqn": "linear.weight", "sparsity_level": 0.5},
]

# define pruner
class SaliencyPruner(BaseStructuredSparsifier):
     """
     Prune filters based on the saliency
     The saliency for a filter is given by the sum of the L1 norms of all of its weights
     """

     def update_mask(self, module, tensor_name, **kwargs):
        # tensor_name will give you the FQN, all other entries in sparse config is present in kwargs
         weights = getattr(module, tensor_name)
         mask = getattr(module.parametrizations, tensor_name)[0].mask

         # use negative weights so we can use topk (we prune out the smallest)
         saliency = -weights.norm(dim=tuple(range(1, weights.dim())), p=1)
         num_to_pick = int(len(mask) * kwargs["sparsity_level"])
         prune = saliency.topk(num_to_pick).indices

         # Set the mask to be false for the rows we want to prune
         mask.data[prune] = False

# define patterns
patterns = _get_default_structured_pruning_patterns()

original = Model()
# any configs passed in here are defaults that are propagated
pruner = SaliencyPruner({"prune_bias": True}, patterns=patterns)
pruner.prepare(original, sparse_config)
pruner.enable_mask_update = True
pruner.step()
pruned_model = pruner.prune()
```

Let's walk through this line by line to understand what the different parts are doing.

###  Pruning Config

To specify the layers to prune we just need the fully qualified name (FQN) of the tensor you are looking to prune in the module.
You can get the FQN of a tensor by printing out `model.named_parameters()`.

For example, for the model defined above, let's say that we want to prune the first Linear layer (Alpha)

To do this we would define a config that looks like this:

``` python
pruning_config = [
    {"tensor_fqn": "seq.0.weight"}
]
```

To prune multiple layers, we just append entries to the pruning config.
**tensor_fqn** is the only required key in the pruning config. You can pass additional information in the config, for example the sparsity level you want to prune to by adding a key to the config. You can then access this additional information when you update the masks.

``` python
pruning_config = [
    {"tensor_fqn": "seq.0.weight", "sparsity_level": 0.7},
    {"tensor_fqn": "seq.2.weight", "sparsity_level": 0.7},
    {"tensor_fqn": "seq.4.weight", "sparsity_level": 0.7},
    {"tensor_fqn": "linear.weight", "sparsity_level": 0.7},
]
```

### Implementing a Pruner

Now that we've defined the sparse config for our model, we need to specify the pruning criteria that we will use.
To do this, we need to extend a `BaseStructuredSparsifier` with a custom `update_mask` function.

This `update_mask` function contains the user logic for picking what weights to prune.

One common pruning criteria is to use the **saliency** of a row, which is defined as the sum of all the L1 norms of the weights in the row.
The idea is to remove the weights that are small, since they wouldn't contribute much to the final prediction.

Below we can see an implemented Saliency Pruner

```python
class SaliencyPruner(BaseStructuredSparsifier):
     """
     Prune filters based on the saliency
     The saliency for a filter is given by the sum of the L1 norms of all of its weights
     """

     def update_mask(self, module, tensor_name, **kwargs):
        # tensor_name will give you the FQN, all other keys in pruning config are present in kwargs
         weights = getattr(module, tensor_name)
         mask = getattr(module.parametrizations, tensor_name)[0].mask

         # use negative weights so we can use topk (we prune out the smallest)
         saliency = -weights.norm(dim=tuple(range(1, weights.dim())), p=1)
         num_to_pick = int(len(mask) * kwargs["sparsity_level"])
         prune = saliency.topk(num_to_pick).indices

         # Set the mask to be false for the rows we want to prune
         mask.data[prune] = False

```

Now that we've defined our pruning config and pruning criteria, we can move on to actually pruning our model.

### Using the Pruner

First, we create an instance of the pruner that we want. Any config settings that are passed in to the Pruner will be propogated to all of the elements in the config. In this case, we will prune a layer by 50%, unless a custom sparsity_level is set in the config.

We can also pass in the pruning pattern dict, which described how to resize different patterns of modules.

```python
# any configs passed in here are defaults that are propagated
pruner = SaliencyPruner({"sparsity_level": 0.5}, patterns=patterns)
```

Next we call `prepare`, which will attach `FakeStructuredSparsity` parameterizations to the tensors specified in the config. These parameterizations will zero out the appropriate weights in order to make the model behave as if it has been pruned.

```python
pruner.prepare(model, sparse_config)
```

However, if we were to run the prepared model currently, we'd see the exact same output as before the prepared model.

This is because we have yet to update the mask, and by default the initial mask is all set to `True` and prunes no rows.
We can calculate the mask given the current weights by takeing a pruning step.

```python
# need to enable mask update
pruner.enable_mask_update = True
pruner.step()
```

After this step is called, the masks will be updated to prune the corresponding weights. You can expect the ouput of the model after this step to be the same as the pruned model.

However, there will be no performance / memory benefits for the model, since the pruned parameters are still present but just zeroed out. In order to achieve performance gains, we need to resize the weight tensors, by calling `prune`.
```python
# to prune the module
pruned_model = pruner.prune()
```

Prune will symbolically trace the model in order to get a graph, and then search that graph for particular patterns (op sequences)

Once prune identifies an op sequence, it is able to resize the weights by calling the pruning function for a paticular block. This is defined by a dictionary that maps patterns (tuple of nn modules / functions) to a particular pruning function. You can extend this pattern dictionary to add support for pruning different modules / patterns.

For this particular example, we would expect to match the `(nn.Linear, nn.ReLU, nn.Linear)` pattern in our default pattern dict.

Once this is done for all matches, `prune` returns your pruned model:

```
# original model
seq.0.weight        | 500, 700        |    350000
seq.0.bias          | 500             |       500
seq.2.weight        | 800, 500        |    400000
seq.4.weight        | 600, 800        |    480000
seq.4.bias          | 600             |       600
linear.weight       | 4, 600          |      2400
=== Total Number of Parameters: 1233500 ===
```
```
# pruned model
seq.0.weight        | 250, 700        |    175000
seq.0.bias          | 250             |       250
seq.2.weight        | 400, 250        |    100000
seq.4.weight        | 300, 400        |    120000
seq.4.bias          | 300             |       300
linear.weight       | 2, 300          |       600
=== Total Number of Parameters: 396150 ===
```

Note that since we prune 50% of the output rows, this also corresponds to pruning 50% of the input columns for the subsequent layer.
This is why our total number of parameters is roughly (1-0.5)* (1-0.5) = 0.25 of the original number of parameters.

## Writing Custom Patterns and Pruning Functions for Structured Pruning
What should you do if the layers you are trying to prune don't match the existing patterns?

If you're working with linear/conv2d layers, it's very probable that you just need to add an entry to the pattern dict mapping your pattern to an existing prune_function.

This is because there are many modules, for example **pooling** that behave the same way and do not need to be modified by the pruning code.

```python
from torch.ao.pruning._experimental.pruner.prune_functions improt prune_conv2d_activation_conv2d

def prune_conv2d_pool_activation_conv2d(
    c1: nn.Conv2d,
    pool: nn.Module,
    activation: Optional[Callable[[Tensor], Tensor]],
    c2: nn.Conv2d,
) -> None:
    prune_conv2d_activation_conv2d(c1, activation, c2)

# note how the pattern defined in the key will be passed to the pruning function as args
my_patterns = {(nn.Conv2d, nn.MaxPool2d, nn.ReLU, nn.Conv2d): prune_conv2d_activation_conv2d}

pruning_patterns = _get_default_structured_pruning_patterns()
pruning_patterns.update(my_patterns)

pruner = SaliencyPruner({}, patterns=pruning_patterns)
```
However, there are also modules like batch norm, which will not work properly without being pruned as well. In this instance, you would need to write a custom pruning function in order to handle that logic properly.

You can see the implemented pruning functions [here](https://github.com/pytorch/pytorch/blob/master/torch/ao/pruning/_experimental/pruner/prune_functions.py) for examples. Please feel free to open a PR so we get a complete set of the patterns that people use.

