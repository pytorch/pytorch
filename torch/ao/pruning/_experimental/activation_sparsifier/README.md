# Activation Sparsifier

## Introduction
Activation sparsifier attaches itself to a layer(s) in the model and prunes the activations passing through them. **Note that the layer weights are not pruned here.**

## How does it work?
The idea is to compute a mask to prune the activations. To compute the mask, we need a representative tensor that generalizes activations coming from all the batches in the dataset.

There are 3 main steps involved:
1. **Aggregation**: The activations coming from inputs across all the batches are aggregated using a user-defined `aggregate_fn`.
A simple example is the add function.
2. **Reduce**: The aggregated activations are then reduced using a user-defined `reduce_fn`. A simple example is average.
3. **Masking**: The reduced activations are then passed into a user-defined `mask_fn` to compute the mask.

Essentially, the high level idea of computing the mask is

```
>>> aggregated_tensor = aggregate_fn([activation for activation in all_activations])
>>> reduced_tensor = reduce_fn(aggregated_tensor)
>>> mask = mask_fn(reduced_tensor)
```

*The activation sparsifier also supports per-feature/channel sparsity. This means that a desired set of features in an activation can be also pruned. The mask will be stored per feature.*

```
>>> # when features = None, mask is a tensor computed on the entire activation tensor
>>> # otherwise, mask is a list of tensors of length = len(features), computed on each feature of activations
>>>
>>> # On a high level, this is how the mask is computed if features is not None
>>> for i in range(len(features)):
>>>    aggregated_tensor_feature = aggregate_fn([activation[features[i]] for activation in all_activations])
>>>    mask[i] = mask_fn(reduce_fn(aggregated_tensor_feature))
```

## Implementation Details
The activation sparsifier attaches itself to a set of layers in a model and then attempts to sparsify the activations flowing through them. *Attach* means registering a [`forward_pre_hook()`](https://pytorch.org/docs/stable/_modules/torch/nn/modules/module.html#register_forward_pre_hook) to the layer.

Let's go over the 3 steps again -
1. **Aggregation**: The activation of aggregation happens by attaching a hook to the layer that specifically applies and stores the aggregated data. The aggregation happens per feature, if the features are specified, otherwise it happens on the entire tensor.
The `aggregate_fn` should accept two input tensors and return an aggregated tensor. Example:
```
def aggregate_fn(tensor1, tensor2):
    return tensor1 + tensor2
```

2. **Reduce**: This is initiated once the `step()` is called. The `reduce_fn()` is called on the aggregated tensor. The goal is to squash the aggregated tensor.
The `reduce_fn` should accept one tensor as argument and return a reduced tensor. Example:
```
def reduce_fn(agg_tensor):
    return agg_tensor.mean(dim=0)
```

3. **Masking**: The computation of the mask happens immediately after the reduce operation. The `mask_fn()` is applied on the reduced tensor. Again, this happens per-feature, if the features are specified.
The `mask_fn` should accept a tensor (reduced) and sparse config as arguments and return a mask (computed using tensor according to the config). Example:
```
def mask_fn(tensor, threshold):  # threshold is the sparse config here
    mask = torch.ones_like(tensor)
    mask[torch.abs(tensor) < threshold] = 0.0
    return mask
```

## API Design
`ActivationSparsifier`: Attaches itself to a model layer and sparsifies the activation flowing through that layer. The user can pass in the default `aggregate_fn`, `reduce_fn` and `mask_fn`. Additionally, `features` and `feature_dim` are also accepted.

`register_layer`: Registers a layer for sparsification. Specifically, registers `forward_pre_hook()` that performs aggregation.

`step`: For each registered layer, applies the `reduce_fn` on aggregated activations and then applies `mask_fn` after reduce operation.

`squash_mask`: Unregisters aggregate hook that was applied earlier and registers sparsification hooks if `attach_sparsify_hook=True`. Sparsification hooks applies the computed mask to the activations before it flows into the registered layer.

## Example

```
# Fetch model
model = SomeModel()

# define some aggregate, reduce and mask functions
def aggregate_fn(tensor1, tensor2):
    return tensor1 + tensor2

def reduce_fn(tensor):
    return tensor.mean(dim=0)

def mask_fn(data, threshold):
    mask = torch.ones_like(tensor)
    mask[torch.abs(tensor) < threshold] = 0.0
    return mask)

# sparse config
default_sparse_config = {"threshold": 0.5}

# define activation sparsifier
act_sparsifier = ActivationSparsifier(model=model, aggregate_fn=aggregate_fn, reduce_fn=reduce_fn, mask_fn=mask_fn, **threshold)

# register some layer to sparsify their activations
act_sparsifier.register_layer(model.some_layer, threshold=0.8)  # custom sparse config

for epoch in range(EPOCHS):
    for input, target in dataset:
        ...
        out = model(input)
        ...
    act_sparsifier.step()  # mask is computed

act_sparsifier.squash_mask(attach_sparsify_hook=True)  # activations are multiplied with the computed mask before flowing through the layer
```
