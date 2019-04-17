import torch
import torch.nn as nn
import copy, math
import torch.nn.functional as F
from torch.autograd import Variable
from .module import Module
from .activation import MultiheadAttention

def get_clones(module, N):
	return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def buildTransformerModel(src_vocab, tgt_vocab, d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6, d_ff=2048, dropout=0.1):
	encoder_layer = TransformerEncoderLayer(d_model, nhead, d_ff, dropout)
	src_embed = Embeddings(d_model, src_vocab) 
	pos_encoder = PositionalEncoding(d_model, dropout) 
	encoder_norm = Tensor1DNorm(d_model) 
	encoder = TransformerEncoder(encoder_layer, num_encoder_layers, src_embed, pos_encoder, encoder_norm)

	decoder_layer = TransformerDecoderLayer(d_model, nhead, d_ff, dropout)	
	tgt_embed = Embeddings(d_model, tgt_vocab) 
	pos_decoder = PositionalEncoding(d_model, dropout) 
	decoder_norm = Tensor1DNorm(d_model) 
	decoder = TransformerDecoder(decoder_layer, num_decoder_layers, tgt_embed, pos_decoder, decoder_norm)

	generator = Generator(d_model, tgt_vocab)
	model = TransformerBase(encoder, decoder, generator)

	for p in model.parameters():
		if p.dim() > 1:
			nn.init.xavier_uniform_(p)
	return model


class TransformerBase(Module):
	"""
	A base class for transformer. The transformer is based on a standard
	Encoder-Decoder architecture.
	Docs comming...
	"""
	def __init__(self,  encoder, decoder, generator = None):
		super(TransformerBase, self).__init__()
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


class TransformerEncoder(Module):
	"""
	TransformerEncoder is a stack of N layers
	Docs comming...
	"""
	def __init__(self, encoder_layer, num_layers, src_embed=None, pos_encoder=None, norm=None):
		super(TransformerEncoder, self).__init__()
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


class TransformerDecoder(Module):
	"""
	TransformerDecoder is a stack of N layers
	Docs comming...
	"""
	def __init__(self, decoder_layer, num_layers, tgt_embed=None, pos_encoder=None, norm=None):
		super(TransformerDecoder, self).__init__()
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

class TransformerEncoderLayer(Module):
	"""
	TransformerEncoder layer is made up of self-attn and feed forward (defined below)
	Docs comming...
	"""
	def __init__(self, d_model, nhead, d_ff=2048,dropout=0.1):
		super(TransformerEncoderLayer, self).__init__()
		self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
		self.ff = FeedForward(d_model, d_ff=d_ff, dropout=dropout)
		self.norm1 = Tensor1DNorm(d_model)
		self.norm2 = Tensor1DNorm(d_model)
		self.dropout1 = nn.Dropout(dropout)
		self.dropout2 = nn.Dropout(dropout)

	def forward(self, x, mask):
		x2 = self.norm1(x)
		x = x + self.dropout1(self.self_attn(x2, x2, x2, mask=None))
		x2 = self.norm2(x)
		x = x + self.dropout2(self.ff(x2))
		return x


class TransformerDecoderLayer(Module):
	"""
	TransformerDecoder layer is made up of self-attn, multihead_attn and feed forward (defined below)
	Docs comming...
	"""
	def __init__(self, d_model, nhead, d_ff=2048, dropout=0.1):
		super(TransformerDecoderLayer, self).__init__()
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
		x = x + self.dropout2(self.multihead_attn(x2, memory, memory))
		x2 = self.norm3(x)
		x = x + self.dropout3(self.ff(x2))		
		return x


# Temporarily leave Tensor1DNorm module here. Will bemoved somewhere else.
class Tensor1DNorm(Module):
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
class FeedForward(Module):
	def __init__(self, d_model, d_ff=2048, dropout = 0.1):
		super(FeedForward, self).__init__() 
		# We set d_ff as a default to 2048
		self.linear1 = nn.Linear(d_model, d_ff)
		self.dropout = nn.Dropout(dropout)
		self.linear2 = nn.Linear(d_ff, d_model)
	
	def forward(self, x):
		return self.linear2(self.dropout(F.relu(self.linear1(x))))


# Temporarily leave Embeddings module here. Will be moved somewhere else.
class Embeddings(Module):
	def __init__(self, d_model, vocab):
		super(Embeddings, self).__init__()
		self.lut = nn.Embedding(vocab, d_model)
		self.d_model = d_model

	def forward(self, x):
		return self.lut(x) * math.sqrt(self.d_model)


# Temporarily leave PositionalEncoding module here. Will be moved somewhere else.
class PositionalEncoding(Module):
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
class Generator(Module):
	"Define standard linear + softmax generation step."
	def __init__(self, d_model, vocab):
		super(Generator, self).__init__()
		self.proj = nn.Linear(d_model, vocab)

	def forward(self, x):
		return F.log_softmax(self.proj(x), dim=-1)


