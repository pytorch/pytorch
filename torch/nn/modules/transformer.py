import torch
import torch.nn as nn
import copy, math
import torch.nn.functional as F
from torch.autograd import Variable

def get_clones(module, N):
	return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class Transformer(nn.Module):
	"""
	A base class for transformer. The transformer is based on a standard
	Encoder-Decoder architecture.
	"""
	def __init__(self,  encoder, decoder, generator = None):
		super(Transformer, self).__init__()
		self.encoder = encoder 
		self.decoder = decoder
		self.generator = generator 

	def forward(self, src, tgt, src_mask=None, tgt_mask=None):
		"Take in and process masked src and target sequences."
		memory = self.encode(src, src_mask)
		output = self.decode(tgt, memory, tgt_mask, src_mask)
		return output
	
	def encode(self, src, src_mask=None):
		return self.encoder(src, src_mask)
	
	def decode(self, tgt, memory, tgt_mask=None, memory_mask=None):
		return self.decoder(tgt, memory, tgt_mask, memory_mask)


class Encoder(nn.Module):
	"Encoder is a stack of N layers"
	def __init__(self, encoder_layer, num_layers, src_embed=None, pos_encoder=None, norm=None):
		super(Encoder, self).__init__()
		self.layers = get_clones(encoder_layer, num_layers)
		self.num_layers = num_layers
		self.src_embed = src_embed
		self.pos_encoder = pos_encoder
		self.norm = norm
		
	def forward(self, src, mask):
		"Pass the input (and mask) through each layer in turn."
		output = src
		if self.src_embed:
			output = self.src_embed(output);

		if self.pos_encoder:
			output = self.pos_encoder(output)
	
		for i in range(self.num_layers):
			output = self.layers[i](output, mask)

		if self.norm:
			output = self.norm(output)

		return output


class Decoder(nn.Module):
	"Decoder is a stack of N layers"
	def __init__(self, decoder_layer, num_layers, tgt_embed=None, pos_encoder=None, norm=None):
		super(Decoder, self).__init__()
		self.layers = get_clones(decoder_layer, num_layers)
		self.num_layers = num_layers
		self.tgt_embed = tgt_embed
		self.pos_encoder = pos_encoder
		self.norm = norm
		
	def forward(self, tgt, memory, tgt_mask, memory_mask):
		"Pass the input (and mask) through each layer in turn."
		output = tgt
		if self.tgt_embed:
			output = self.tgt_embed(output);

		if self.pos_encoder:
			output = self.pos_encoder(output)
	
		for i in range(self.num_layers):
			output = self.layers[i](output, memory, tgt_mask, memory_mask)

		if self.norm:
			output = self.norm(output)

		return output

class EncoderLayer(nn.Module):
	"Encoder layer is made up of self-attn and feed forward (defined below)"
	def __init__(self, d_model, nhead, d_ff=2048,dropout=0.1):
		super(EncoderLayer, self).__init__()
		self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
		self.ff = FeedForward(d_model, d_ff=d_ff, dropout=dropout)
		self.norm1 = Tensor1DNorm(d_model)
		self.norm2 = Tensor1DNorm(d_model)
		self.dropout1 = nn.Dropout(dropout)
		self.dropout2 = nn.Dropout(dropout)

	def forward(self, x, mask):
		x2 = self.norm1(x)
		x = x + self.dropout1(self.self_attn(x2,x2,x2,mask))
		x2 = self.norm2(x)
		x = x + self.dropout2(self.ff(x2))
		return x


class DecoderLayer(nn.Module):
	"Decoder layer is made up of self-attn, multihead_attn and feed forward (defined below)"
	def __init__(self, d_model, nhead, d_ff=2048, dropout=0.1):
		super(DecoderLayer, self).__init__()
		self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
		self.multihead_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
		self.ff = FeedForward(d_model, d_ff=d_ff, dropout=dropout)
		self.norm1 = Tensor1DNorm(d_model)
		self.norm2 = Tensor1DNorm(d_model)
		self.norm3 = Tensor1DNorm(d_model)
		self.dropout1 = nn.Dropout(dropout)
		self.dropout2 = nn.Dropout(dropout)
		self.dropout3 = nn.Dropout(dropout)

	def forward(self, x, memory, tgt_mask, src_mask):
		x2 = self.norm1(x)
		x = x + self.dropout1(self.self_attn(x2, x2, x2, tgt_mask))
		x2 = self.norm2(x)
		x = x + self.dropout2(self.multihead_attn(x2, memory, memory, src_mask))
		x2 = self.norm3(x)
		x = x + self.dropout3(self.ff(x2))		
		return x

# Temporarily leave Tensor1DNorm module here. Will bemoved somewhere else.
class Tensor1DNorm(nn.Module):
	"Normalization component with two learnable variables"
	def __init__(self, d_model, eps = 1e-6):
		super(Tensor1DNorm, self).__init__()
		self.size = d_model
		# create two learnable parameters to calibrate normalisation
		self.alpha = nn.Parameter(torch.ones(self.size))
		self.bias = nn.Parameter(torch.zeros(self.size))
		self.eps = eps
	
	def forward(self, x):
		x_mean = x.mean(dim=-1, keepdim=True)
		x_std = x.std(dim=-1, keepdim=True)
		return self.alpha * (x - x_mean) / (x_std + self.eps) + self.bias

# Temporarily leave FeedForward module here. Will be moved somewhere else.
class FeedForward(nn.Module):
	def __init__(self, d_model, d_ff=2048, dropout = 0.1):
		super(FeedForward, self).__init__() 
		# We set d_ff as a default to 2048
		self.linear1 = nn.Linear(d_model, d_ff)
		self.dropout = nn.Dropout(dropout)
		self.linear2 = nn.Linear(d_ff, d_model)
	
	def forward(self, x):
		return self.linear2(self.dropout(F.relu(self.linear1(x))))


# Temporarily leave Embeddings module here. Will be moved somewhere else.
class Embeddings(nn.Module):
	def __init__(self, d_model, vocab):
		super(Embeddings, self).__init__()
		self.lut = nn.Embedding(vocab, d_model)
		self.d_model = d_model

	def forward(self, x):
		return self.lut(x) * math.sqrt(self.d_model)


# Temporarily leave PositionalEncoding module here. Will be moved somewhere else.
class PositionalEncoding(nn.Module):
	"Implement the PE function."
	def __init__(self, d_model, dropout, max_len=5000):
		super(PositionalEncoding, self).__init__()
		self.dropout = nn.Dropout(p=dropout)
		
		# Compute the positional encodings once in log space.
		pe = torch.zeros(max_len, d_model)
		position = torch.arange(0, max_len, dtype = torch.float).unsqueeze(1)
		div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
		pe[:, 0::2] = torch.sin(position * div_term)
		pe[:, 1::2] = torch.cos(position * div_term)
		pe = pe.unsqueeze(0)
		self.register_buffer('pe', pe)
		
	def forward(self, x):
		x = x + Variable(self.pe[:, :x.size(1)], 
						 requires_grad=False)
		return self.dropout(x)


# Temporarily leave Generator module here. Will be moved somewhere else.
class Generator(nn.Module):
    "Define standard linear + softmax generation step."
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)


# Temporarily leave MultiheadAttention here. Will be removed once the MultiheadAttention PR landed.
def attention(query, key, value, mask=None, dropout=None):
	"Compute 'Scaled Dot Product Attention'"
	d_k = query.size(-1)
	scores = torch.matmul(query, key.transpose(-2, -1)) \
			 / math.sqrt(d_k)
	if mask is not None:
		scores = scores.masked_fill(mask == 0, -1e9)
	p_attn = F.softmax(scores, dim = -1)
	if dropout is not None:
		p_attn = dropout(p_attn)
	return torch.matmul(p_attn, value), p_attn


class MultiheadAttention(nn.Module):
	def __init__(self, d_model, h, dropout=0.1):
		"Take in model size and number of heads."
		super(MultiheadAttention, self).__init__()
		assert d_model % h == 0
		# We assume d_v always equals d_k
		self.d_k = d_model // h
		self.h = h
		self.linears = get_clones(nn.Linear(d_model, d_model), 4)
		self.attn = None
		self.dropout = nn.Dropout(p=dropout)
		
	def forward(self, query, key, value, mask=None):
		"Implements Figure 2"
		if mask is not None:
			# Same mask applied to all h heads.
			mask = mask.unsqueeze(1)
		nbatches = query.size(0)
		
		# 1) Do all the linear projections in batch from d_model => h x d_k 
		query, key, value = \
			[l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
			 for l, x in zip(self.linears, (query, key, value))]
		
		# 2) Apply attention on all the projected vectors in batch. 
		x, self.attn = attention(query, key, value, mask=mask, 
								 dropout=self.dropout)
		
		# 3) "Concat" using a view and apply a final linear. 
		x = x.transpose(1, 2).contiguous() \
			 .view(nbatches, -1, self.h * self.d_k)
		return self.linears[-1](x)


