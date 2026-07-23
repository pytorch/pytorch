# Data Sparsifier
## Intro
The data sparsifier inherits from the `BaseSparsifier` class. It attempts to sparsify data tensors in general (trainable and non-trainable).

## Implementation Details
The data sparsifier does not receive a model or a layer to sparsify. Hence, the mask needs to be owned by the data sparsifier. This is achieved by introducing a private container model that registers the data as a parametrized buffer.

The BaseDataSparsifier handles all the housekeeping while allowing the user to just implement the `update_mask` logic in their implementation.

## Supported data
1. torch tensors (torch.Tensor)
2. parameters (nn.Parameter)
3. embedding and embedding bags (nn.Embeddings / nn.EmbeddingBag)

## API details
`BaseDataSparsifier`: base class with abstract method `update_mask` that computes the new mask for all the data.

`add_data`: Accepts name, data tuple and registers the data as a parametrized buffer inside the container model. Note that the data is always associated to a name. A custom sparse config can be provided along with the name, data pair. If not provided, the default config will be applied while doing the sparsification.
If the named data already exists, then it is replaced with the new data. The config and mask will be retained for the new data unless not specified to.
To not the old mask, set `reuse_mask=False`. If the `config` is explicitly passed in, it will be updated.

**Note**: name containing '.' is not a valid name for the data sparsifier

```
data_sparsifier = ImplementedDataSparsifier()
data_sparsifier.add_data(name=name, data=data, **some_config)
```

`step`: applies the update_mask() logic to all the data.

```
data_sparsifier.step()
```

`get_mask`: retrieves the mask given the name of the data.

`get_data`: retrieves the data given the `name` argument. Accepts additional argument `return_original` which when set to `True` does not apply the mask while returning
the data tensor. Example:

```
original_data = data_sparsifier.get_data(name=name, return_original=True)  # returns data with no mask applied
sparsified_data = data_sparsifier.get_data(name=name, return_original=False)  # returns data * mask
```

`squash_mask`: removes the parametrizations on the data and applies mask to the data when `leave_parametrized=True`.Also, accepts list of strings to squash mask for. If none, squashes mask for all the keys.
```
data_sparsifier.squash_mask()
```

`state_dict`: Returns dictionary that can be serialized.

## Write your own data sparsifier.
The custom data sparsifier should be inherited from the BaseDataSparsifier class and the `update_mask()` should be implemented. For example, the following data sparsifier zeros out all entries of the tensor smaller than some threshold value.

```
class ImplementedDataSparsifier(BaseDataSparsifier):
    def __init__(self, threshold):
        super().__init__(threshold=threshold)

    def update_mask(self, name, data, threshold):
        mask = self.get_mask(name)
        mask[torch.abs(data) < threshold] = 0.0
```

## Using Data Sparsifier
### Simple example

```
tensor1 = torch.randn(100, 100)
param1 = nn.Parameter(torch.randn(200, 32))

my_sparsifier = ImplementedDataSparsifier(threshold=0.2)
my_sparsifier.add_data(name='tensor1', data=tensor1, threshold=0.5)
my_sparsifier.add_data(name='param1', data=param1)

my_sparsifier.step()  # computes mask

my_sparsifier.squash_mask()  # applies and removes mask
```

### Sparsifying model embeddings

```
class Model(nn.Module):
    def __init__(self, feature_dim, emb_dim, num_classes):
        self.emb = nn.EmbeddingBag(feature_dim, emb_dim)
        self.linear1 = nn.Linear(emb_dim, 32)
        self.linear2 = nn.Linear(32, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.emb(x)
        out = self.relu(self.linear1(out))
        out = self.linear2(out)
        return out

model = Model(100, 32, 10)
my_sparsifier = ImplementedDataSparsifier(threshold=0.5)
my_sparsifier.add_data(name='emb', data=model.emb)

...
# Train model
...

my_sparsifier.step()  # creates mask for embeddings

my_sparsifier.squash_mask()  # applies and removes mask
```

### Using in the context of training data
Sometimes if the input data can be sparsified before sending it to the model, then we can do so by using the data sparsifier.

The batched input data needs to be attached to the data sparsified before sending it to the model.

```
model = SomeModel()

data_sparsifier = ImplementedDataSparsifier(threshold=0.2)

data_name = 'train_data'

for x, y in train_data_loader:
    x = data_sparsifier.add_data(name=data_name, data=x)
    ...
    y_out = model(x)
    ...
    data_sparsifier.step()

```


**Note**:
1. It is the responsibility of the `BaseDataSparsifier` to call the `self.update_mask` when appropriate.
2. The mask should be modified in place.

    Some valid inplace operations are:
    1. Change a portion of a mask: `mask[:10] = torch.zeros(10)`
    2. Use an inplace operator: `mask *= another_mask`
    3. Change the underlying data: `mask.data = torch.zeros_like(mask)`

    Non-inplace operations are not valid, and might lead to bugs. For example:

    1. Reassignment of a mask: `mask = torch.zeros_like(mask)`
    2. Non-inplace arithmetic operations: `mask = mask * another_mask`
3. Data sparsifier `name` argument cannot have a '.' in it.
