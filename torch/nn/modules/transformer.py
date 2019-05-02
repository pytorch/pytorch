import torch
import copy
import math
from .. import functional as F
from torch.autograd import Variable
from .module import Module
from .activation import MultiheadAttention
from .container import ModuleList
import numpy as np
from ..parameter import Parameter
from ..init import xavier_uniform_
from .dropout import Dropout
from .linear import Linear
from .sparse import Embedding

class TransformerBase(Module):
    r"""A base Transformer class. The transformer is based on a standard
        Encoder-Decoder architecture.

    Args:
        encoder: an encoder component (required).
        decoder: a decoder component (required).
        generator: a linear network model (optional).

    Examples:
        >>> transformer_model = nn.TransformerBase(encoder, decoder)
    """

    def __init__(self, encoder, decoder, generator=None):
        super(TransformerBase, self).__init__()
        self.encoder = encoder 
        self.decoder = decoder
        self.generator = generator 


class Transformer(TransformerBase):
    r"""A transformer model applied for sequence-to-sequence transform. 
        User is able to modified the attributes as needed.

    Args:
        src_vocab: the number of vocabularies in the source sequence (required). 
        tgt_vocab: the number of vocabularies in the target sequence (required). 
        d_model: the dimension of the encoder/decoder embedding models (default=512).
        nhead: the number of heads in the multiheadattention models (default=8).
        num_encoder_layers: the number of sub-encoder-layers in the encoder (default=6).
        num_decoder_layers: the number of sub-decoder-layers in the decoder (default=6).
        d_ff: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).

    Examples::
        >>> transformer_model = nn.Transformer(src_vocab, tgt_vocab)
        >>> transformer_model = nn.Transformer(src_vocab, tgt_vocab, nhead=16, num_encoder_layers=12)
    """

    def __init__(self, src_vocab, tgt_vocab, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, d_ff=2048, dropout=0.1):
        encoder_layer = TransformerEncoderLayer(d_model, nhead, d_ff, dropout)
        src_embed = Embeddings(d_model, src_vocab) 
        pos_encoder = PositionalEncoding(d_model, dropout) 
        encoder_norm = LayerNorm(d_model) 
        encoder = TransformerEncoder(encoder_layer, num_encoder_layers, src_embed, pos_encoder, encoder_norm)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, d_ff, dropout)    
        tgt_embed = Embeddings(d_model, tgt_vocab) 
        pos_decoder = PositionalEncoding(d_model, dropout) 
        decoder_norm = LayerNorm(d_model) 
        decoder = TransformerDecoder(decoder_layer, num_decoder_layers, tgt_embed, pos_decoder, decoder_norm)

        generator = Generator(d_model, tgt_vocab)

        super(Transformer, self).__init__(encoder, decoder, generator)

        self._reset_parameters()

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None):
        r"""Take in and process masked source/target sequences.

        Args:
            src: the sequence to the encoder (required).
            tgt: the sequence to the decoder (required).
            src_mask: the mask for the src sequence (optional).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the encoder output (optional).

        Shape:
            src: [source sequence length, batch size]
            tgt: [target sequence length, batch size]
            src_mask: [source sequence length, source sequence length]
            tgt_mask: [target sequence length, target sequence length]
            memory_mask: [target sequence length, source sequence length]
            Note: The maksed positions are filled with float('-inf'). 
                  Unmasked positions are filled with float(0.0). Masks ensure that the predictions 
                  for position i depend only on the information before position i.

            output: [target sequence length, batch size, tgt_vocab]
            Note: Due to the multi-head attention architecture in the transformer model, 
                  the output sequence length of a transformer is same as the input sequence
                  (i.e. target) length of the decode. 

        Examples:
            >>> output = transformer_model(src, tgt, src_mask=src_mask, tgt_mask=tgt_mask)
        """

        memory = self.encode(src, src_mask)
        output = self.decode(tgt, memory, tgt_mask, memory_mask)

        if self.generator:
            output = self.generator(output)

        return output

    def encode(self, src, src_mask=None):
        return self.encoder(src, src_mask)

    def decode(self, tgt, memory, tgt_mask=None, memory_mask=None):
        return self.decoder(tgt, memory, tgt_mask, memory_mask)

    def generate_square_subsequent_mask(self, sz):
        r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0)
        """
        attn_shape = (sz, sz)
        mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
        mask = torch.from_numpy(mask) == 0
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask    

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""

        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)


class TransformerEncoder(Module):
    r"""TransformerEncoder is a stack of N encoder layers

    Args:
        encoder_layer: an instance of the TransformerEncoderLayer() class (required). 
        num_layers: the number of sub-encoder-layers in the encoder (required).
        src_embed: the embedding model for the source sequence (optional).
        pos_encoder: the positional encoding model for the source sequence (optional).
        norm: the layer normalization component (optional).

    Examples::
        >>> transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
    """

    def __init__(self, encoder_layer, num_layers, src_embed=None, pos_encoder=None, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.src_embed = src_embed
        self.pos_encoder = pos_encoder
        self.norm = norm

    def forward(self, src, mask=None):
        r"""Pass the input through the endocder layers in turn.

        Args:
            src: the sequnce to the encoder (required).
            src_mask: the mask for the src sequence (optional).

        Shape:
            see the docs in Transformer class.
        """
        output = src
        if self.src_embed:
            output = self.src_embed(output)

        if self.pos_encoder:
            output = self.pos_encoder(output)

        for i in range(self.num_layers):
            output = self.layers[i](output, mask)

        if self.norm:
            output = self.norm(output)

        return output


class TransformerDecoder(Module):
    r"""TransformerDecoder is a stack of N decoder layers

    Args:
        decoder_layer: an instance of the TransformerDecoderLayer() class (required). 
        num_layers: the number of sub-decoder-layers in the decoder (required).
        tgt_embed: the embedding model for the target sequence (optional).
        pos_encoder: the positional encoding model for the target sequence (optional).
        norm: the layer normalization component (optional).

    Examples::
        >>> transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers)
    """

    def __init__(self, decoder_layer, num_layers, tgt_embed=None, pos_encoder=None, norm=None):
        super(TransformerDecoder, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.tgt_embed = tgt_embed
        self.pos_encoder = pos_encoder
        self.norm = norm

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        r"""Pass the inputs (and mask) through the decoder layer in turn.

        Args:
            tgt: the sequence to the decoder (required).
            memory: the sequnce from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).

        Shape:
            see the docs in Transformer class.
        """
        output = tgt
        if self.tgt_embed:
            output = self.tgt_embed(output)

        if self.pos_encoder:
            output = self.pos_encoder(output)

        for i in range(self.num_layers):
            output = self.layers[i](output, memory, tgt_mask, memory_mask)

        if self.norm:
            output = self.norm(output)

        return output

class TransformerEncoderLayer(Module):
    r"""TransformerEncoderLayer is made up of self-attn and feedforward network.

    Args:
        d_model: the embed dim (required).
        nhead: the number of heads in the multiheadattention models (required).
        d_ff: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model, nhead)
    """

    def __init__(self, d_model, nhead, d_ff=2048, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.ff = FeedForward(d_model, d_ff=d_ff, dropout=dropout)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

    def forward(self, src, src_mask=None):
        r"""Pass the input through the endocder layer.

        Args:
            src: the sequnce to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).

        Shape:
            see the docs in Transformer class.
        """
        src2 = self.norm1(src)
        src = src + self.dropout1(self.self_attn(src2, src2, src2, attn_mask=src_mask)[0])
        src2 = self.norm2(src)
        src = src + self.dropout2(self.ff(src2))
        return src


class TransformerDecoderLayer(Module):
    r"""TransformerDecoderLayer is made up of self-attn, multi-head-attn and feedforward network.

    Args:
        d_model: the embed dim (required).
        nhead: the number of heads in the multiheadattention models (required).
        d_ff: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).

    Examples::
        >>> decoder_layer = nn.TransformerDecoderLayer(d_model, nhead)
    """

    def __init__(self, d_model, nhead, d_ff=2048, dropout=0.1):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.ff = FeedForward(d_model, d_ff=d_ff, dropout=dropout)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        r"""Pass the inputs (and mask) through the decoder layer.

        Args:
            tgt: the sequence to the decoder layer (required).
            memory: the sequnce from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).

        Shape:
            see the docs in Transformer class.
        """
        tgt2 = self.norm1(tgt)
        tgt = tgt + self.dropout1(self.self_attn(tgt2, tgt2, tgt2, attn_mask=tgt_mask)[0])
        tgt2 = self.norm2(tgt)
        tgt = tgt + self.dropout2(self.multihead_attn(tgt2, memory, memory, attn_mask=memory_mask)[0])
        tgt2 = self.norm3(tgt)
        tgt = tgt + self.dropout3(self.ff(tgt2))        
        return tgt


# Temporarily leave LayerNorm module here. Will be moved somewhere else.
class LayerNorm(Module):
    r"""Normalize the activities of the neurons to reduce the training time. Computing the mean and 
        variance used for normalization from all of the summed inputs 
        to the neurons in a layer on a single training case.

    Args:
        d_model: the embed dim of the tensors fed to the LayerNorm model (required).

    Examples:
        >>> layer_norm = LayerNorm(d_model)
    """

    def __init__(self, d_model, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.size = d_model
        # create two learnable parameters to calibrate normalisation
        self.alpha = Parameter(torch.ones(self.size))
        self.bias = Parameter(torch.zeros(self.size))
        self.eps = eps

    def forward(self, x):
        r"""Inputs of forward function

        Args:
            x: the tensor fed to the LayerNorm model (required).

        Shape:
            x: [sequence length, batch size, embed dim]

            output: [sequence length, batch size, embed dim]    

        Examples:
            >>> output = layer_norm(x)
        """
        x_mean = x.mean(dim=-1, keepdim=True)
        x_std = x.std(dim=-1, keepdim=True)
        return self.alpha * (x - x_mean) / (x_std + self.eps) + self.bias


# Temporarily leave FeedForward module here. Will be moved somewhere else.
class FeedForward(Module):
    r"""A fully connected feed-forward network, which consists of two 
        linear transformations with a ReLU activation in between.

    Args:
        d_model: the embed dim (required).
        d_ff: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).

    Examples::
        >>> ff = FeedForward(d_model)
    """

    def __init__(self, d_model, d_ff=2048, dropout=0.1):
        super(FeedForward, self).__init__() 
        self.linear1 = Linear(d_model, d_ff)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(d_ff, d_model)

    def forward(self, x):
        r"""Inputs of forward function

        Args:
            x: the tensor fed to FeedForward model (required).

        Shape:
            x: [sequence length, batch size, embed dim]

            output: [sequence length, batch size, embed dim]    

        Examples:
            >>> output = ff(x)
        """
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


# Temporarily leave Embeddings module here. Will be moved somewhere else.
class Embeddings(Module):
    r"""Sequence embedding. Normalized by math.sqrt(embed dim).

    Args:
        d_model: the embed dim (required).
        vocab: the number of vocabularies in the sequence (required). 

    Examples:
        >>> embed_model = Embeddings(d_model, vocab)
    """
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        r"""Inputs of forward function

        Args:
            x: the sequence fed to the embedding model (required).

        Shape:
            x: [sequence length, batch size]

            output: [sequence length, batch size, embed dim]    

        Examples:
            >>> output = embed_model(x)
        """

        return self.lut(x) * math.sqrt(self.d_model)


# Temporarily leave PositionalEncoding module here. Will be moved somewhere else.
class PositionalEncoding(Module):
    r"""Inject some information about the relative or absolute position of the tokens 
        in the sequence. The positional encodings have the same dimension as 
        the embeddings, so that the two can be summed. Here, we use sine and cosine 
        functions of different frequencies.

    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)

    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).

    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function

        Args:
            x: the sequence fed to the positional encoder model (required).

        Shape:
            x: [sequence length, batch size, embed dim]

            output: [sequence length, batch size, embed dim]    

        Examples:
            >>> output = pos_encoder(x)
        """

        x = x + Variable(self.pe[:x.size(0), :], 
                         requires_grad=False)
        return self.dropout(x)


# Temporarily leave Generator module here. Will be moved somewhere else.
class Generator(Module):
    r"""A generator processing the output of the decoder. It convertes sequence 
        tensors from embedding to vocabs. log_softmax function is attached to
        the end.

    Args:
        d_model: the embed dim (required).
        vocab: the number of vocabularies in the target sequence (required). 

    Examples:
        >>> generator = Generator(d_model, vocab)
    """

    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = Linear(d_model, vocab)

    def forward(self, x):
        r"""Inputs of forward function

        Args:
            x: the sequence fed to the generator model (required).

        Shape:
            x: [sequence length, batch size, embed dim]

            output: [sequence length, batch size, vocab]    

        Examples:
            >>> output = generator(x)
        """

        return F.log_softmax(self.proj(x), dim=-1)


def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])
