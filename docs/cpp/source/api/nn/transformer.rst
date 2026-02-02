Transformer Layers
==================

Transformer layers use self-attention mechanisms to process sequences in parallel,
enabling efficient training on long sequences. They are the foundation of modern
NLP models (BERT, GPT) and increasingly used in vision and other domains.

- **Transformer**: Complete encoder-decoder architecture
- **TransformerEncoder/Decoder**: Standalone encoder or decoder stacks
- **TransformerEncoderLayer/DecoderLayer**: Individual transformer blocks
- **MultiheadAttention**: Core attention mechanism used throughout

**Key parameters:**

- ``d_model``: Dimension of the model (embedding dimension)
- ``nhead``: Number of attention heads
- ``num_encoder_layers/num_decoder_layers``: Number of stacked layers
- ``dim_feedforward``: Dimension of feedforward network
- ``dropout``: Dropout rate for regularization

Transformer
-----------

Complete encoder-decoder transformer architecture.

.. doxygenclass:: torch::nn::Transformer
   :members:
   :undoc-members:

.. doxygenclass:: torch::nn::TransformerImpl
   :members:
   :undoc-members:

**Example:**

.. code-block:: cpp

   auto transformer = torch::nn::Transformer(
       torch::nn::TransformerOptions()
           .d_model(512)
           .nhead(8)
           .num_encoder_layers(6)
           .num_decoder_layers(6)
           .dim_feedforward(2048)
           .dropout(0.1));

TransformerEncoder
------------------

Stack of encoder layers for processing source sequences.

.. doxygenclass:: torch::nn::TransformerEncoder
   :members:
   :undoc-members:

.. doxygenclass:: torch::nn::TransformerEncoderImpl
   :members:
   :undoc-members:

TransformerDecoder
------------------

Stack of decoder layers for generating target sequences.

.. doxygenclass:: torch::nn::TransformerDecoder
   :members:
   :undoc-members:

.. doxygenclass:: torch::nn::TransformerDecoderImpl
   :members:
   :undoc-members:

TransformerEncoderLayer
-----------------------

Single encoder layer with self-attention and feedforward network.

.. doxygenclass:: torch::nn::TransformerEncoderLayerImpl
   :members:
   :undoc-members:

TransformerDecoderLayer
-----------------------

Single decoder layer with self-attention, cross-attention, and feedforward network.

.. doxygenclass:: TransformerDecoderLayerImpl
   :members:
   :undoc-members:

MultiheadAttention
------------------

Scaled dot-product attention with multiple parallel heads.

.. doxygenclass:: torch::nn::MultiheadAttention
   :members:
   :undoc-members:

.. doxygenclass:: torch::nn::MultiheadAttentionImpl
   :members:
   :undoc-members:
