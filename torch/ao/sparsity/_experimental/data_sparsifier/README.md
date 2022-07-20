# Data Sparsifier
## Intro
The data sparsifier inherits from the ```BaseSparsifier``` class.
The current base sparsifier accepts a model and sparsifies its layers by introducing parametrization.
<br>
The data sparsifier attempts to sparsify data tensors in general. The data tensors can also be weight tensors of the model. The base sparsifier can be generalized into data sparsifier in the case when weight tensors of the model layers are passed instead of the layer.

## Implementation Details
The base sparsifier introduces sparsity in the model by introducing layer parametrizations which means that the mask is owned by the model and not the sparsifier class.

The data sparsifier does not receive a model or a layer to sparsify. Hence, the mask needs to be owned by the data sparsifier. This is acheived by introducing a private container model that registers the data as a parametrized buffer.

The BaseDataSparsifier handles all the housekeeping while allowing the user to just implement the ```update_mask``` logic in their implementation.

## Supported data
1. torch tensors (torch.Tensor)
2. parameters (nn.Parameter)
3. embedding and embedding bags (nn.Embeddings / nn.EmbeddingBag)

## API details
```BaseDataSparsifier```: base class with abstract method ```update_mask``` that computes the new mask for all the data.

```add_data```: Accepts name, data tuple and registers the data as a parametrized buffer inside the container model. Note that the data is always associated to a name. A custom sparse config can be provided along with the name, data pair. If not provided, the default config will be applied while doing the sparsification.

```
data_sparsifier = ImplementedDataSparsifier()
data_sparsifier.add_data(name=name, data=data, **some_config)
```
```step```: applies the update_mask() logic to all the data.

```
data_sparsifier.step()
```
```get_mask```: retrieves the mask given the name of the data.

```squash_mask```: removes the parametrizations on the data and applies mask to the data when ```leave_parametrized=True```.Also, accepts list of strings to squash mask for. If none, squashes mask for all the keys.
```
data_sparsifier.squash_mask()
```

## Write your own data sparsifier.
The custom data sparsifier should be inherited from the BaseDataSparsifier class and the ```update_mask()``` should be implemented. For example, the following data sparsifier just creates a mask to zero out the first row of the data.
```
class ImplementedDataSparsifier(BaseDataSparsifier):
    def update_mask(self, name, data, **kwargs):
        mask = self.get_mask(name)
        mask[0] = 0
```

Note::
1. It is the responsibility of the ```BaseDataSparsifier``` to call the ```self.update_mask``` when appropriate.
2. The mask should be modified in place.

    Some valid inplace operations are:
    1. Change a portion of a mask: ```mask[:10] = torch.zeros(10)```
    2. Use an inplace operator: ```mask *= another_mask```
    3. Change the underlying data: ```mask.data = torch.zeros_like(mask)```

    Non-inplace operations are not valid, and might lead to bugs. For example:

    1. Reassignment of a mask: ```mask = torch.zeros_like(mask)```
    2. Non-inplace arithmetic operations: ```mask = mask * another_mask```
