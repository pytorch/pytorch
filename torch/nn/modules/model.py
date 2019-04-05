import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
import matplotlib.pyplot as plt
#from transformer import EncoderLayer, DecoderLayer, Tensor1DNorm, Encoder, Decoder, Transformer, Embeddings, PositionalEncoding, Generator, MultiheadAttention, FeedForward

from transformer import EncoderLayer, DecoderLayer, Tensor1DNorm, Encoder, Decoder, Transformer, Embeddings, PositionalEncoding, Generator
def buildTransformerModel(src_vocab, tgt_vocab, d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6, d_ff=2048, dropout=0.1):
	encoder_layer = EncoderLayer(d_model, nhead, d_ff, dropout)
	src_embed = Embeddings(d_model, src_vocab) 
	pos_encoder = PositionalEncoding(d_model, dropout) 
	encoder_norm = Tensor1DNorm(d_model) 
	encoder = Encoder(encoder_layer, num_encoder_layers, src_embed, pos_encoder, encoder_norm)

	decoder_layer = DecoderLayer(d_model, nhead, d_ff, dropout)	
	tgt_embed = Embeddings(d_model, tgt_vocab) 
	pos_decoder = PositionalEncoding(d_model, dropout) 
	decoder_norm = Tensor1DNorm(d_model) 
	decoder = Decoder(decoder_layer, num_decoder_layers, tgt_embed, pos_decoder, decoder_norm)

	generator = Generator(d_model, tgt_vocab)
	model = Transformer(encoder, decoder, generator)

	# This was important from their code. 
	# Initialize parameters with Glorot / fan_avg.
	for p in model.parameters():
		if p.dim() > 1:
			nn.init.xavier_uniform_(p)
	return model

def subsequent_mask(size):
	"Mask out subsequent positions."
	attn_shape = (1, size, size)
	subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
	return torch.from_numpy(subsequent_mask) == 0


class Batch:
	"Object for holding a batch of data with mask during training."
	def __init__(self, src, trg=None, pad=0):
		self.src = src
		self.src_mask = (src != pad).unsqueeze(-2)
		if trg is not None:
			self.trg = trg[:, :-1]
			self.trg_y = trg[:, 1:]
			self.trg_mask = \
				self.make_std_mask(self.trg, pad)
			self.ntokens = (self.trg_y != pad).data.sum().item()
	
	@staticmethod
	def make_std_mask(tgt, pad):
		"Create a mask to hide padding and future words."
		tgt_mask = (tgt != pad).unsqueeze(-2)
		tgt_mask = tgt_mask & Variable(
			subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
		return tgt_mask


def run_epoch(data_iter, model, loss_compute):
	"Standard Training and Logging Function"
	start = time.time()
	total_tokens = 0
	total_loss = 0
	tokens = 0
	for i, batch in enumerate(data_iter):
		out = model.forward(batch.src, batch.trg, 
							batch.src_mask, batch.trg_mask)
		loss = loss_compute(out, batch.trg_y, batch.ntokens)
		total_loss += loss
		total_tokens += batch.ntokens
		tokens += batch.ntokens
		if i % 5 == 0:
			elapsed = time.time() - start
			print("Epoch Step: %d Loss: %f Tokens per Sec: %f" %
					(i, loss / batch.ntokens, tokens / elapsed))
			start = time.time()
			tokens = 0
	return total_loss / total_tokens


global max_src_in_batch, max_tgt_in_batch
def batch_size_fn(new, count, sofar):
	"Keep augmenting batch and calculate total number of tokens + padding."
	global max_src_in_batch, max_tgt_in_batch
	if count == 1:
		max_src_in_batch = 0
		max_tgt_in_batch = 0
	max_src_in_batch = max(max_src_in_batch,  len(new.src))
	max_tgt_in_batch = max(max_tgt_in_batch,  len(new.trg) + 2)
	src_elements = count * max_src_in_batch
	tgt_elements = count * max_tgt_in_batch
	return max(src_elements, tgt_elements)


class NoamOpt:
	"Optim wrapper that implements rate."
	def __init__(self, model_size, factor, warmup, optimizer):
		self.optimizer = optimizer
		self._step = 0
		self.warmup = warmup
		self.factor = factor
		self.model_size = model_size
		self._rate = 0
		
	def step(self):
		"Update parameters and rate"
		self._step += 1
		rate = self.rate()
		for p in self.optimizer.param_groups:
			p['lr'] = rate
		self._rate = rate
		self.optimizer.step()
		
	def rate(self, step = None):
		"Implement `lrate` above"
		if step is None:
			step = self._step
		return self.factor * \
			(self.model_size ** (-0.5) *
			min(step ** (-0.5), step * self.warmup ** (-1.5)))
		
def get_std_opt(model):
	return NoamOpt(model.src_embed[0].d_model, 2, 4000,
			torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))


# Three settings of the lrate hyperparameters.
opts = [NoamOpt(512, 1, 4000, None), 
		NoamOpt(512, 1, 8000, None),
		NoamOpt(256, 1, 4000, None)]
plt.plot(np.arange(1, 20000), [[opt.rate(i) for opt in opts] for i in range(1, 20000)])
plt.legend(["512:4000", "512:8000", "256:4000"])


class LabelSmoothing(nn.Module):
	"Implement label smoothing."
	def __init__(self, size, padding_idx, smoothing=0.0):
		super(LabelSmoothing, self).__init__()
		self.criterion = nn.KLDivLoss(size_average=False)
		self.padding_idx = padding_idx
		self.confidence = 1.0 - smoothing
		self.smoothing = smoothing
		self.size = size
		self.true_dist = None
		
	def forward(self, x, target):
		assert x.size(1) == self.size
		true_dist = x.data.clone()
		true_dist.fill_(self.smoothing / (self.size - 2))
		true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
		true_dist[:, self.padding_idx] = 0
		mask = torch.nonzero(target.data == self.padding_idx)
		if mask.dim() > 0:
			true_dist.index_fill_(0, mask.squeeze(), 0.0)
		self.true_dist = true_dist
		return self.criterion(x, Variable(true_dist, requires_grad=False))

######################################################################
# First example

def data_gen(V, batch, nbatches):
	"Generate random data for a src-tgt copy task."
	for i in range(nbatches):
		data = torch.from_numpy(np.random.randint(1, V, size=(batch, 10)))
		data[:, 0] = 1
		src = Variable(data, requires_grad=False)
		tgt = Variable(data, requires_grad=False)
		yield Batch(src, tgt, 0)


class SimpleLossCompute:
	"A simple loss compute and train function."
	def __init__(self, generator, criterion, opt=None):
		self.generator = generator
		self.criterion = criterion
		self.opt = opt
		
	def __call__(self, x, y, norm):
		x = self.generator(x)
		loss = self.criterion(x.contiguous().view(-1, x.size(-1)), 
							  y.contiguous().view(-1)) / norm
		loss.backward()
		if self.opt is not None:
			self.opt.step()
			self.opt.optimizer.zero_grad()
	   # return loss.data[0] * norm
		return loss.item() * norm


# Train the simple copy task.
V = 11
criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.0)

model = buildTransformerModel(V, V, num_encoder_layers=2, num_decoder_layers=2)
model_opt = NoamOpt(model.encoder.src_embed.d_model, 1, 400,
		torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

for epoch in range(10):
	print("-------------------------------------------")
	print("Epoch %d:" %epoch)
	model.train()
	run_epoch(data_gen(V, 50, 21), model, 
			  SimpleLossCompute(model.generator, criterion, model_opt))
	model.eval()
	print("evaluation result:")
	run_epoch(data_gen(V, 50, 6), model, 
					SimpleLossCompute(model.generator, criterion, None))


def greedy_decode(model, src, src_mask, max_len, start_symbol):
	memory = model.encode(src, src_mask)
	ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
	for i in range(max_len-1):
		out = model.decode(Variable(ys), memory, 
						   Variable(subsequent_mask(ys.size(1)).type_as(src.data)),
						   src_mask) 
		prob = model.generator(out[:, -1])
		_, next_word = torch.max(prob, dim = 1)
		next_word = next_word.data[0]
		ys = torch.cat([ys, 
						torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
	return ys

model.eval()
src = Variable(torch.LongTensor([[1,2,3,4,5,6,7,8,9,10]]) )
src_mask = Variable(torch.ones(1, 1, 10) )
print(greedy_decode(model, src, src_mask, max_len=10, start_symbol=1))


