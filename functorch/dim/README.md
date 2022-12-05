Named Tensors using First-class Dimensions in PyTorch
=====================================================

-- Zachary DeVito [@Zachary_DeVito](https://twitter.com/Zachary_DeVito)

_An implementation of [named tensors](https://namedtensor.github.io) with the functionality of [einsum](http://einops.rocks]http://einops.rocks) , batching ([vmap](https://jax.readthedocs.io/en/latest/jax.html#vectorization-vmap), [xmap](https://jax.readthedocs.io/en/latest/notebooks/xmap_tutorial.html)), and tensor indexing by adding dimension objects to PyTorch_.

The tensor input to a resnet might have the shape [8, 3, 224, 224] but informally we think of those dimensions as 'batch', 'channel', 'width', and 'height'. Eventhough 'width' and 'height' have the same _size_ we still think of them as separate dimensions, and if we have two _different_ images, we think of both as sharing the _same_ 'channel' dimension.

Named tensors gives these dimensions names. [PyTorch's current implementation](https://pytorch.org/docs/stable/named_tensor.html) uses strings to name dimensions. Instead, this library introduces a Python object, a `Dim`, to represent the concept. By expanding the semantics of tensors with dim objects, in addition to naming dimensions, we can get behavior equivalent to batching transforms (xmap, vmap), einops-style rearrangement, and loop-style tensor indexing.

A preview:

```py
from torchdim import dims

# einsum
def mm(A: torch.Tensor, B: torch.Tensor):
    i, j, k = dims(3)
    r = (A[i, k] * B[k, j]).sum(k)
    return r.order(i, j)

# rearrange
def pixel_shuffle(img: torch.Tensor, upscale_factor=2):
    h2, w2, c, b, h, w = dims(6)
    h2.size = w2.size = upscale_factor
    return img[b, (c, h2, w2), h, w].order(b, c, (h, h2), (w, w2))

# batching
def bmm(A: torch.Tensor, B: torch.Tensor):
    i = dims(1)
    return mm(A[i], B[i]).order(i)

# indexing
def embedding_bag(input: torch.Tensor, embedding_weights: torch.Tensor):
    batch, sequence, features = dims(3)
    r = embedding_weights[input[batch, sequence], features].sum(sequence)
    return r.order(batch, features)
```

Installation
============


_torchdim is a preview release so that we can collect feedback on the API. It may have bugs, and there are known places where performance can be improved._

First-class dims are a library that extends PyTorch, so they need to be installed separately.
We may eventually upstream them into PyTorch itself along with `functorch`.


We have to install a nightly build of PyTorch so first set up an environment:

```sh
conda create --name dim
conda activate dim
```

First-class dims requires a fairly recent nightly build of PyTorch so that functorch will work. You can install it using one of these commands:

```sh
# For CUDA 10.2
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch-nightly
# For CUDA 11.3
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch-nightly
# For CPU-only build
conda install pytorch torchvision torchaudio cpuonly -c pytorch-nightly
```

Install dim. You will be asked for github credentials to access the fairinternal organization.

```sh
pip install ninja  # Makes the build go faster
pip install --user "git+https://github.com/facebookresearch/torchdim"
```

Creating and Binding Dims
=========================

Python objects that represent dimension are created using the `dims` operator.[^1]

```py
import torch
from torchdim import dims

batch, channel, width, height = dims(4)
```

The existing implementation of [Named Tensors](https://pytorch.org/docs/stable/named_tensor.html) in PyTorch, or [JAX's xmap](https://jax.readthedocs.io/en/latest/notebooks/xmap_tutorial.html) use strings to name dimensions. We call these dimensions _first class_ because they are Python objects.

In addition to the normal _positional_ dimensions in a tensor, tensors can also have a separate set of first-class dimensions.

You can create tensors with first-class dimensions by indexing the normal positional dimensions of a tensor with a dimension object. The `ndim` property continues to list the number of positional dimensions, while the new `dims` property lists all the bound first-class dimensions.

```py
input = torch.rand(2, 3, 224, 224)
print(input.ndim)
> 4

input_fc = input[batch, channel, width, height]
print(input_fc.dims) # first class dimensions
> (batch, channel, width, height)


# since we converted all the positional dimensions
# first class `input_fc` has 0 positional dimensions now.
print(input_fc.ndim)
> 0
```

Notice that indexing creates a _new_ Tensor, `input_fc` with bound first-class dimensions. It does not modify the original tensor `input`, which still has 4 positional dimensions.

```py
print(input.ndim) # unchanged
> 4
```

Importantly, indexing with square brackets _applies only to positional dimensions_, so attempting to index a tensor with only first class dims will error[^2]:

```py
try:
    input_fc[0]
except ValueError as ve:
    print(ve)
> at least 1 indices were supplied but the tensor only has 0 dimensions
```

Generally, it is possible to construct tensors with a mixture of positional and first class dimensions:

```py
input_mixed = input[batch, :, :, height]
print(input_mixed.dims)
> (batch, height)

print(input_mixed.ndim)
> 2
```

Dimension Sizes
---------------

Dimensions will take on the size of the first thing they are bound to:

```py
input = torch.rand(3)
x = dims(1)
input_fc = input[x]
print(x.size)
> 3
```

But you can also directly set the size of dimension:

```py
i = dims(1)

i.size = 5 # ok, i previously did not have a size

i.size = 5 # ok, it already had the size 5
try:
    i.size = 3
except Exception as e:
    print(e)
> Dim 'i' previously bound to a dimension of size 5 cannot bind to a dimension of size 3

j = dims(sizes=[4]) # can also be set on construction
```

[^1]: We use a bit of Python introspection to set the debug names for the dimensions based on the names of the variables they are assigned to.
[^2]: Indexing of first-class dimensions can be done with the `index` method by specifying the dimension to be index into (e.g. `input_fc.index(batch, 0)`.

Semantics of Dimensions
=======================
The power of named tensors arises from how the first-class dimensions in the Tensors composed with existing operations.

Three rules define how dimension objects behave with existing Tensors.

Rule 1: Implicit Batching
-------------------------
**Tensor operations (e.g. `input + bias`) are implicitly batched over the union of the first-class dimensions in their inputs.**

If `input` has dimensions `batch, channel` and `bias` has dimension `channel`, the output will have the union of those dimensions (`batch, channel`), and the result will computed as if there was a loop over all the first-class dimensions.[^3]

```py
input_positional = torch.rand(128, 32)
bias_positional = torch.rand(32)

batch, channel = dims(2)
input = input_positional[batch, channel]
bias = bias_positional[channel]

result = input + bias
print(result.dims)
> (batch, channel)
```

It is helpful think of operators on tensors with first-class dimensions by analogy to code with explicit loops over dimensions, with the first-class dimensions of the inputs acting as implicit `for` loops, and the values in the tensor being scalars within the body of the loop:

```py
# mental model: loop-level analogy
for batch in range(batch.size):
    for channel in range(channel.size):
        input = input_positional[batch, channels]
        bias = bias_positional[channels]
        result[batch, channels] =  input + bias # arithmetic on scalars
```

Positional dimensions behave as they did before (e.g. for + they will broadcast), and can be thought of as being a standard tensor _used within the implicit loops_ defined by first-class dimensions.

In this example, we broke down the expression into lines that bind the dimension to positional tensors and then another line to do the compute. In practice, we often combine these in one statement:

```py
result = input_positional[batch, channel] + bias_positional[channel]
result.dims
```

[^3] This rule is similar to how named dimensions in xmap behave within a function, but instead of introducing the dimensions via a functional transform, they are bound on the objects using indexing.


Rule 2: Specifying dimensions
-----------------------------
**Wherever an integer is used to specify a dimension in the existing torch operator, a first-class dimensions can be used instead to tell the operator to work over that dimension.**

```py
batch, channel, width, height = dims(4)
input_positional = torch.rand(2, 3, 224, 224)
input = input_positional[batch, channel, width, height]
avg_pixel_color = input.mean((width, height))

print(avg_pixel_color.dims)
> (batch, channel)
```

Any other first-class dimensions (e.g. batch, channel) are still implicitly batched according to Rule #1.

Rule 3: Dims are Tensors
------------------------
**A first-class dimension `d` can be used wherever a Tensor is expected. It will act as if it were a tensor whose only dimension is itself, `d`, and the values along the dimension are the indices of each entry `(0, 1, 2, ..., d.size - 1)`**

```py
print(channel.dims)
> (channel,)

print(channel + 1000)
> tensor([1000, 1001, 1002])
> with dims=(channel,) sizes=(3,)
```

This means that a dimension used as a tensor acts as an index into that dimension. Going back to our loop-level analogy, it is analogous to using the loop variable as a value:

```py
# mental model: loop-level analogy
for channel in range(batch.size):
    result[channel] = channel + 1000
```

Arithmetic using dimension indices comes up a lot, such as the mask for an upper triangular part of a matrix. Using dims as tensors makes it easy:

```py
from torchdim import dims
i, j = dims(sizes=[4, 4])
print(i <= j)
> tensor([[ True,  True,  True,  True],
>         [False,  True,  True,  True],
>         [False, False,  True,  True],
>         [False, False, False,  True]])
> with dims=(i, j) sizes=(4, 4)
```

Because of the intentional similarity to loop-level code, using dimensions as tensors makes complicated indexing arithmetic easier to read.

Here is code that lookups up features in an embedding table given a sequence of ids:

```py
sequence, features = dims(2)
embeddings = torch.rand(8, 128)
words = torch.tensor([5, 4, 0,])

state = embeddings[words[sequence], features]
print(state.dims)
> (sequence, features)
```

With the following analogy to loops:

```py
# mental model: loop-level analogy

for sequence in range(words.size(0)):
    for features in range(embeddings.size(1)):
        state = embeddings[words[sequence], features]
```

Earlier we showed how binding tensors dimension is done with indexing `A[i, j]`. In fact, this binding is just the normal indexing operator. Its behavior follows directly from the behavior of indexing with tensor indices combined with Rule #3 and Rule #1. The expression `A[i + 1, j]` also creates a tensor with dimensions `i` and `j` but with different indexing math. The implementation knows when simple indexing patterns are used and only actually runs a kernel to do indexing when needed.

Unbinding Dims
-------------
The `order` method converts first-class dimensions in a tensor back to normal positional dimensions by specifying an order for those dimensions.[^4]

By specifying a different order from how things were originally bound, it is easy to do transpositions.

```py
i, j = dims(2)
A = torch.rand(3, 4)
A_T = A[i, j].order(j, i)
assert torch.allclose(A.T, A_T)
```

Indexing acts left-to-right, and `order` also places the new dimensions back on the left, so it possible to work on tensors that have mixed positional and first-class dimensions:

```py
B = torch.rand(3, 4, 5)
B_T = B[i, j].order(j, i)
assert torch.allclose(B.permute(1, 0, 2), B_T)
```

[^4] `order` is actually just a synonym for the already-existing `permute` method, which takes a list a dimension specifiers and puts the tensor in that order because rule #2 says that first-class dims can be passed as arguments to functions that previously took only integers as dimensions. However, the name `permute` is confusing in this context since it implies dim objects have an original order, so we prefer to use `order` when writing code.

Flattening and Splitting Dims
-----------------------------

**Tuples of dimensions** can be passed to both indexing and `order`. In indexing, this will split the dimension being indexed across the dimensions in the tuple.  In `order` it will flatten the dimensions in a single positional dimension:

```py
i, j, k = dims(3)
j.size = 2
A = torch.rand(6, 4)
a = A[(i, j), k] # split dim 0 into i,j
print(i.size, j.size, k.size)
> 3 2 4

r = a.order(i, (j, k)) # flatten j and k
print(r.shape)
> torch.Size([3, 8])
```

The size of one unsized dimension in a tuple such as `i` can be inferred if the other sizes are known.

Examples
========

The usefulness of dimension objects is best seen through examples. Let's look at some different ways they can be used.

Einsum-style Products
---------------------
Rather than having [einsum](https://pytorch.org/docs/stable/generated/torch.einsum.html) as a custom operator, it is possible to express matrix products directly as a composition of multiplies and summations. The implementation will pattern match any multiplication followed by a sum to the right matrix-multiply operator.

```py
def mm(A, B):
    i, j, k = dims(3)
    r = (A[i, k] * B[k, j]).sum(k)
    return r.order(i, j)
mm(torch.rand(3, 4), torch.rand(4, 5)).shape
```

The implementation of named tensors delays the execution of multiply to see if a summation follows it as it does above. If so, it will turn this pattern into the correct _optimized matrix product_, similar to how the `einsum` function works.

Since it is no longer necessary to manually match math to matrix functions, other tensor products are easier to express, like the Gram matrix used in style transfer:

```py
def gram_matrix_new(y):
    b, c, c2, h, w = dims()
    r = (y[b, c, h, w] * y[b, c2, h, w]).sum((h, w))
    r = r / (h.size * w.size)
    return r.order(b, c, c2)

gram_matrix_new(torch.rand(1, 2, 3, 4))
# [example adapted from http://einops.rocks/pytorch-examples.html]
```

Attention is another example that has several matrix products embedded inside it:

```py
from torchdim import softmax
def attention(K, Q, V):
    batch, channel, key, query = dims(4)
    k = K[batch, channel, key]
    q = Q[batch, channel, query]
    v = V[batch, channel, key]

    a = (k * q).sum(channel) # matrix multiply
    a = softmax(a * (channel.size ** -0.5), dim=key)
    r = (v * a).sum(key) # matrix multiply
    return torch.cat((r.order(batch, channel, query), Q), dim=1)

inputs = (torch.rand(2, 3, 4) for _ in range(3))
attention(*inputs)
# [example adapted from http://einops.rocks/pytorch-examples.html]
```

Reshaping tensors (einops)
--------------------------

Lots of operations in deep learning are just different ways of reshaping, splitting, and joining dimensions, such as the pixel shuffle used to upscale an image by turning channels into pixels:

```py
def pixel_shuffle(img, upscale_factor=2):
    h2, w2, c, b, h, w = dims(6)
    h2.size = w2.size = upscale_factor
    return img[b, (c, h2, w2), h, w].order(b, c, (h, h2), (w, w2))
```

[Einops](http://einops.rocks) is an extension to einsum that adds support for the manipulation of dimensions through a few custom operators such as `rearrange`:

```py
def pixel_shuffle_einops(img, upscale_factor=2):
    from einops import rearrange
    return rearrange(img, 'b (c h2 w2) h w -> b c (h h2) (w w2)', h2=upscale_factor, w2=upscale_factor)
```

Named tensors with first-class dimensions can accomplish the same goal, but using PyTorch's existing operator set.

Automatically batching Code (`vmap`, `xmap`)
-----------------------------

The implicit batching of Rule #1 means it is easy to created batched versions of existing PyTorch code. Simply bind a dim to the dimensions that should act as a batch, and then pass the tensor to the unbatched function. Since the unbatched function does not know about the dim, the dim will be implicitly batched over:

```py
batch_size, feature_size = 3, 5
weights = torch.randn(feature_size)

def model(feature_vec):
    # Very simple linear model with activation
    assert feature_vec.dim() == 1
    return feature_vec.dot(weights).relu()

examples = torch.randn(batch_size, feature_size)
batch = dims(1)
r = model(examples[batch])
print(r)
# in functorch: result = functorch.vmap(model)(examples)
> tensor([0.4775, 0.0000, 0.3423])
> with dims=(batch,) sizes=(3,)
```

This pattern also composes well with other code that also uses first class dimensions. For instance, we can write batched matrix multiply `bmm` by batching the `mm` operator.

It doesn't matter whether the implementation of the function uses dimension objects, it is also possible to add additional batch dimensions and then call a function:

```py
def bmm(A, B):
    i = dims(1) # note: i here is a different value from i inside mm so it works
    return mm(A[i], B[i]).order(i)
```

The equivalent code in JAX, using [xmap or vmap](https://jax.readthedocs.io/en/latest/notebooks/quickstart.html#auto-vectorization-with-vmap) are transforms over functions. So there is a lot of syntactic distance between the specification of the dimension mappings, and the values where those mappings apply. Dims express the mapping as indexing of the tensor, right at the place where the function is being applied.


[xmap examples](https://jax.readthedocs.io/en/latest/notebooks/xmap_tutorial.html):

```py
in_axes = [['inputs', 'hidden', ...],
           ['hidden', 'classes', ...],
           ['batch', 'inputs', ...],
           ['batch', ...]]

loss = xmap(named_loss, in_axes=in_axes, out_axes=[...])
print(loss(w1, w2, images, labels))
```

Equivalent with dimension objects:

```py
batch, inputs, hidden, classes = dims(4)
print(loss(w1[inputs, hidden], w2[hidden, classes], images[batch, inputs], labels[batch],
      batch, inputs, hidden, classes))
```


Composing matrix products, reshaping, and batching:
---------------------

Multi-headed attention is a good example of how these different uses compose. It reshapes the inputs, splitting out different attention heads. It batches over those attention heads, and it uses matrix products to compute attention scores.

```py
from torchdim import softmax
def multiheadattention(q, k, v, num_attention_heads, dropout_prob, use_positional_embedding):
    batch, query_sequence, key_sequence, heads, features = dims(5)
    heads.size = num_attention_heads

    # binding dimensions, and unflattening the heads from the feature dimension
    q = q[batch, query_sequence, [heads, features]]
    k = k[batch, key_sequence, [heads, features]]
    v = v[batch, key_sequence, [heads, features]]

    # einsum-style operators to calculate scores,
    attention_scores = (q*k).sum(features) * (features.size ** -0.5)

    # use first-class dim to specify dimension for softmax
    attention_probs = softmax(attention_scores, dim=key_sequence)

    # dropout work pointwise, following Rule #1
    attention_probs = torch.nn.functional.dropout(attention_probs, p=dropout_prob)

    # another matrix product
    context_layer = (attention_probs*v).sum(key_sequence)

    # flatten heads back into features
    return context_layer.order(batch, query_sequence, [heads, features])
```

Indexing
--------

Rule #3 enables indexing because dimensions act as loop indices when used as a tensor. This allows for a lot of powerful behavior. The simplest might be using the dimensions to compute masks, such as extracting the upper triangular part of a matrix:

```py
from torch import where
def triu(A):
    i,j = dims()
    a = A[i, j]
    return where(i <= j, a, 0).order(i, j)
triu(torch.rand(3, 4))
```

Embedding bag does an embedding table lookup followed by a sum, which can be expressed concisely:

```py
def embedding_bag(input, embedding_weights):
    batch, sequence, features = dims(3)
    r = embedding_weights[input[batch, sequence], features].sum(sequence)
    return r.order(batch, features)

input = torch.tensor([[1, 0, 4, 3]])
W = torch.rand(5,2)
embedding_bag(input, W)
```

Relative positional embeddings associate an embedding vector with the distance between the query and the key in the sequence.
For instance, a key 3 and query 5 will have embedding ID `(5-3)=2`. We can use first-class dimensions to do the indexing arithmetic, and the embedding lookup:

```py
def relative_positional_embedding(q, k, distance_embedding_weight):
    batch, query_sequence, key_sequence, heads, features = dims(5)
    q = q[batch, query_sequence, [heads, features]]
    k = k[batch, key_sequence, [heads, features]]

    distance = query_sequence - key_sequence
    n_embeddings = distance_embedding_weight.size(0)
    index_bias = n_embeddings // 2

    assert key_sequence.size + bias <= n_embeddings

    # indexing with dims
    positional_embedding = distance_embedding_weight[distance + index_bias, features]

    # matrix multiplies with dims
    relative_position_scores_query = (q*positional_embedding).sum(features)
    relative_position_scores_key = (k*positional_embedding).sum(features)
    return  (relative_position_scores_query + relative_position_scores_key).order(batch, heads, key_sequence, query_sequence)
```

Tensor Puzzlers
===============

[Tensor Puzzlers](https://github.com/srush/Tensor-Puzzles), created by Sasha Rush, are a good exercise for learning the numpy and torch APIs by figuring out how to define common operations using a small set of primitive tensor operations.

However, the difficulty of many of the puzzlers lies not in how to compute the answer but the awkwardness of the primitives themselves.

**With first class dimensions, these puzzlers are nearly the same as the spec that defines them**


### Puzzle 3 - outer

Compute [outer](https://numpy.org/doc/stable/reference/generated/numpy.outer.html) - the outer product of two vectors.

```py
def outer_spec(a, b, out):
    for i in range(len(out)):
        for j in range(len(out[0])):
            out[i][j] = a[i] * b[j]

def outer(a, b):
    i, j = dims(2)
    return (a[i] * b[j]).order(i, j)
```

### Puzzle 4 - diag

Compute [diag](https://numpy.org/doc/stable/reference/generated/numpy.diag.html) - the diagonal vector of a square matrix.

```py
def diag_spec(a, out):
    for i in range(len(a)):
        out[i] = a[i][i]

def diag(a):
    i = dims(1)
    return a[i, i].order(i)
```

### Puzzle 5 - eye

Compute [eye](https://numpy.org/doc/stable/reference/generated/numpy.eye.html) - the identity matrix.

```py
from torch import where
def eye_spec(out):
    for i in range(len(out)):
        out[i][i] = 1

def eye(j: int):
    i,j = dims(sizes=[j, j])
    return where(i == j, 1, 0).order(i, j)
```

### Puzzle 6 - triu

Compute [triu](https://numpy.org/doc/stable/reference/generated/numpy.triu.html) - the upper triangular matrix.

```py
def triu_spec(out):
    for i in range(len(out)):
        for j in range(len(out)):
            if i <= j:
                out[i][j] = 1
            else:
                out[i][j] = 0

def triu(j: int):
    i,j = dims(sizes=[j, j])
    return where(i <= j, 1, 0).order(i, j)
```

### Puzzle 8 - diff

Compute [diff](https://numpy.org/doc/stable/reference/generated/numpy.diff.html) - the running difference.

```py
def diff_spec(a, out):
    out[0] = a[0]
    for i in range(1, len(out)):
        out[i] = a[i] - a[i - 1]
def diff(a, i: int):
    i = dims(1)
    d = a[i] - a[i - 1]
    return where(i - 1 >= 0, d, a[i]).order(i)
```

### Puzzle 9 - vstack

Compute [vstack](https://numpy.org/doc/stable/reference/generated/numpy.vstack.html) - the matrix of two vectors

```py
def vstack_spec(a, b, out):
    for i in range(len(out[0])):
        out[0][i] = a[i]
        out[1][i] = b[i]

def vstack(a, b):
    v, i = dims(sizes=[2, None])
    return where(v == 0,  a[i], b[i]).order(v, i)
```

### Puzzle 10 - roll

Compute [roll](https://numpy.org/doc/stable/reference/generated/numpy.roll.html) - the vector shifted 1 circular position.

```py
def roll_spec(a, out):
    for i in range(len(out)):
        if i + 1 < len(out):
            out[i] = a[i + 1]
        else:
            out[i] = a[i + 1 - len(out)]

def roll(a, i: int):
    i = dims(sizes=[a.size(0)])
    return a[where(i + 1 < i.size, i + 1, 0)].order(i)
```

### Puzzle 11 - flip

Compute [flip](https://numpy.org/doc/stable/reference/generated/numpy.flip.html) - the reversed vector

```py
def flip_spec(a, out):
    for i in range(len(out)):
        out[i] = a[len(out) - i - 1]

def flip(a, i: int):
    i = dims(sizes=[a.size(0)])
    return a[i.size - i - 1].order(i)
```

### Puzzle 14 - sequence_mask


Compute [sequence_mask](https://www.tensorflow.org/api_docs/python/tf/sequence_mask) - pad out to length per batch.

```py
def sequence_mask_spec(values, length, out):
    for i in range(len(out)):
        for j in range(len(out[0])):
            if j < length[i]:
                out[i][j] = values[i][j]
            else:
                out[i][j] = 0

def sequence_mask(values, length):
    j, i = dims()
    v = values[i, j]
    return where(j < length[i], v, 0).order(i, j)
```

Advantages of First-class Dimensions over String Dimensions
===================================================================

The most prominent difference between named tensors using first-class dimensions and alternatives (einops, named tensors implemented in PyTorch today , [tensors considered harmful](https://nlp.seas.harvard.edu/NamedTensor), or xmap) is that dimensions are objects rather than strings. Using objects has a number of nice properties.

### Avoiding naming conflicts

Using strings for dimensions introduces the possibility that two unrelated dimensions are given the same name. Using objects instead makes it clear the same names are not the same dimension. It's like the difference between having only global variables, and having the ability to locally bind names in functions.
 For instance, we defined `bmm` by batching a call to `mm`, and even though they both use the name `i` to identify a dimension.  Because each `i` is a different object, there is no naming conflict:

```py
def mm(A, B):
    i, j, k = dims()
    r = (A[i, k] * B[k, j]).sum(k)
    return r.order(i, j)

def bmm(A, B):
    i = dims() # note: doesn't matter than mm internally also uses i
    return mm(A[i], B[i])
```

Einops avoids conflicts by ensuring names are all introduced and removed in a single expression, but this precludes using long-lived dimensions to present implicit batching similar to xmap. When nested, JAX's xmap seems to consider axes the same if the string name matches. In the above example it would consider the `i` dimension to be the same dimension in both `bmm` and `mm` so the code would error.


### Reuse the same operator set

Having a new object type allows us to extend the existing operator set of PyTorch rather than come up with new operators. For instance, binding dimensions using indexing follows semantically from Rules #1 and #3, so there is no need for a special operator to do binding. Even unbinding is just the `permute` operator which follows from Rule #2, though we call it `order` for clarity. In contrast, using strings requires coming up with new APIs such as `einsum` for matrix multiplies, or `rearrange` for doing permutations.

### Allows dims to act as tensors

Rule #3 is not possible with strings since we cannot make strings behave as tensors. Without this rule, all of the indirect indexing that dims enable would not be easy to express.

### Dims can have methods
For instance, as objects, dims can have a size, which allows us to do size inference of dimensions in various places in the API where string based APIs would have to take additional arguments specifying size.


Comparison to tensor compilers or languages (e.g. TVM or Dex)
=============================================================

The semantics and surface syntax of dimension objects resembles the kind of code written in tensor compilers such as [Halide](https://halide-lang.org), [TVM](https://tvm.apache.org), [Tensor Comprehensions](https://github.com/facebookresearch/TensorComprehensions), or the language [Dex](https://github.com/google-research/dex-lang).

These compilers and language have syntax and semantics that resemble the loop-level analogy similar to first-class dimensions. However, as compilers or statically typed languages, they require some binding code to go from running deep learning framework code in Python to using the compiled language. This often at least requires refactoring the compiled parts into their own functions, and may require defining a gradient function. Similar to graph mode frameworks, this adds friction to using and debugging the code.

Dimension objects are just an extension of the existing PyTorch tensors and eager semantics, so there is no friction switching between normal Python code and code that uses them. However, since loops over the dimensions are defined implicitly, they can still execute in Python with good performance compared to explicit loops. Furthermore, with dimension objects, a tensors containing dimensions can compute through code that is oblivious to the dimension such as batching examples. There is no need to separate code into 'compiled' vs 'eager'.

In this way, first-class dims are a way of adapting the nicer syntax of these array compilers and languages to eager numpy-style libraries.


Performance Expectations
========================
First-class dimensions are not a compiler. They provide syntax for existing PyTorch operations such as advanced indexing that is easier to read and write. For large sized tensors, the performance of any statements including them will be the same as using the already existing operations. An important exception is the pattern matching of products and summation, where performance will be improved by issuing to a matrix-multiply kernel. The C++ implementation of dimensions adds a small overhead of around 2us on top of PyTorch's normal overhead of 8us to each function that uses them. In the future, the implementation can encorporate more fusion optimization to further improve performance of this style of code.


## License
Functorch has a BSD-style license, as found in the [LICENSE](LICENSE) file.
