Embedding Layers
================

Embedding layers map discrete tokens (words, categories, IDs) to dense vector
representations. They are the foundation of NLP models and recommendation systems.

- **Embedding**: Standard lookup table that maps indices to dense vectors
- **EmbeddingBag**: Computes sums or means of embeddings (efficient for sparse features)

**Key parameters:**

- ``num_embeddings``: Size of the vocabulary (number of unique tokens)
- ``embedding_dim``: Dimension of each embedding vector
- ``padding_idx``: Index that outputs zeros (useful for padding tokens)

Embedding
---------

.. doxygenclass:: torch::nn::Embedding
   :members:
   :undoc-members:

.. doxygenclass:: torch::nn::EmbeddingImpl
   :members:
   :undoc-members:

**Example:**

.. code-block:: cpp

   auto embedding = torch::nn::Embedding(
       torch::nn::EmbeddingOptions(10000, 256)  // num_embeddings, embedding_dim
           .padding_idx(0));

   auto indices = torch::tensor({1, 2, 3, 4});
   auto embedded = embedding->forward(indices);  // [4, 256]

EmbeddingBag
------------

.. doxygenclass:: torch::nn::EmbeddingBag
   :members:
   :undoc-members:

.. doxygenclass:: torch::nn::EmbeddingBagImpl
   :members:
   :undoc-members:
