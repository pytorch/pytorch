from torch import nn
from torch.nn.utils import rnn as rnn_utils


class RnnModelWithPackedSequence(nn.Module):
    def __init__(self, model, batch_first):
        super(RnnModelWithPackedSequence, self).__init__()
        self.model = model
        self.batch_first = batch_first

    def forward(self, input, *args):
        args, seq_lengths = args[:-1], args[-1]
        input = rnn_utils.pack_padded_sequence(input, seq_lengths, self.batch_first)
        rets = self.model(input, *args)
        ret, rets = rets[0], rets[1:]
        ret, _ = rnn_utils.pad_packed_sequence(ret, self.batch_first)
        return tuple([ret] + list(rets))

class RnnModelWithPackedSequenceWithoutState(nn.Module):
    def __init__(self, model, batch_first):
        super(RnnModelWithPackedSequenceWithoutState, self).__init__()
        self.model = model
        self.batch_first = batch_first

    def forward(self, input, seq_lengths):
        input = rnn_utils.pack_padded_sequence(input, seq_lengths, self.batch_first)
        rets = self.model(input)
        ret, rets = rets[0], rets[1:]
        ret, _ = rnn_utils.pad_packed_sequence(ret, self.batch_first)
        return list([ret] + list(rets))

class RnnModelWithPackedSequenceWithState(nn.Module):
    def __init__(self, model, batch_first):
        super(RnnModelWithPackedSequenceWithState, self).__init__()
        self.model = model
        self.batch_first = batch_first

    def forward(self, input, hx, seq_lengths):
        input = rnn_utils.pack_padded_sequence(input, seq_lengths, self.batch_first)
        rets = self.model(input, hx)
        ret, rets = rets[0], rets[1:]
        ret, _ = rnn_utils.pad_packed_sequence(ret, self.batch_first)
        return list([ret] + list(rets))
