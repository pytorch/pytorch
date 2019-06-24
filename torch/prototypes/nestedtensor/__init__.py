import torch
import nested

torch.nn.functional.embedding = nested.embedding_monkey
torch.nn.functional.dropout = nested.dropout_monkey
torch.nn.functional.cross_entropy = nested.cross_entropy_monkey
torch.nn.functional.linear = nested.linear_monkey
torch.nn.modules.LSTM.forward = nested.nn_lstm_forward_monkey

torch.nestedtensor = nested.make_tensor
