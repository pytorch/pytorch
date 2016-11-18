## Sparse layers
### Embedding

A simple lookup table that stores embeddings of a fixed dictionary and size

```python
# an Embedding module containing 10 tensors of size 3
embedding = nn.Embedding(10, 3)
# a batch of 2 samples of 4 indices each
input = torch.LongTensor([[1,2,4,5],[4,3,2,9]])
print(embedding(input))
# example with padding_idx
embedding = nn.Embedding(10, 3, padding_idx=0)
input = torch.LongTensor([[0,2,0,5]])
print(embedding(input))
```

This module is often used to store word embeddings and retrieve them using indices.
The input to the module is a list of indices, and the output is the corresponding
word embeddings.

#### Constructor Arguments

Parameter | Default | Description
--------- | ------- | -----------
num_embeddings |  | size of the dictionary of embeddings
embedding_dim |  | the size of each embedding vector
padding_idx | None | If given, pads the output with zeros whenever it encounters the index.
max_norm | None | If given, will renormalize the embeddings to always have a norm lesser than this
norm_type |  | The p of the p-norm to compute for the max_norm option
scale_grad_by_freq |  | if given, this will scale gradients by the frequency of the words in the dictionary.

#### Expected Shape
       | Shape | Description 
------ | ----- | ------------
 input | [ *, * ]  | Input is a 2D mini_batch LongTensor of m x n indices to extract from the Embedding dictionary
output | [ * , *, * ]   | Output shape = m x n x embedding_dim
