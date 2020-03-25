import torch
import torch.nn as nn
import torch.nn.functional as F

import torchaudio_models as models

import math
from collections import OrderedDict

from utils import make_functional, load_weights

def get_wav2letter(device):
    N = 10
    input_frames = 700
    vocab_size = 28
    model = models.Wav2Letter(num_classes=vocab_size)
    criterion = torch.nn.NLLLoss()
    model.to(device)
    params, names = make_functional(model)

    inputs = torch.rand([N, 1, input_frames], device=device)
    labels = torch.rand(N, 3, device=device).mul(vocab_size).long()

    def forward(*new_params):
        load_weights(model, names, new_params)
        out = model(inputs)

        loss = criterion(out, labels)
        return loss

    return forward, params

def get_deepspeech(device):
    # Taken from  https://github.com/SeanNaren/deepspeech.pytorch with modifications

    supported_rnns = {
        'lstm': nn.LSTM,
        'rnn': nn.RNN,
        'gru': nn.GRU
    }

    class SequenceWise(nn.Module):
        def __init__(self, module):
            """
            Collapses input of dim T*N*H to (T*N)*H, and applies to a module.
            Allows handling of variable sequence lengths and minibatch sizes.
            :param module: Module to apply input to.
            """
            super(SequenceWise, self).__init__()
            self.module = module

        def forward(self, x):
            t, n = x.size(0), x.size(1)
            x = x.view(t * n, -1)
            x = self.module(x)
            x = x.view(t, n, -1)
            return x

        def __repr__(self):
            tmpstr = self.__class__.__name__ + ' (\n'
            tmpstr += self.module.__repr__()
            tmpstr += ')'
            return tmpstr


    class MaskConv(nn.Module):
        def __init__(self, seq_module):
            """
            Adds padding to the output of the module based on the given lengths. This is to ensure that the
            results of the model do not change when batch sizes change during inference.
            Input needs to be in the shape of (BxCxDxT)
            :param seq_module: The sequential module containing the conv stack.
            """
            super(MaskConv, self).__init__()
            self.seq_module = seq_module

        def forward(self, x, lengths):
            """
            :param x: The input of size BxCxDxT
            :param lengths: The actual length of each sequence in the batch
            :return: Masked output from the module
            """
            for module in self.seq_module:
                x = module(x)
                mask = torch.BoolTensor(x.size()).fill_(0)
                if x.is_cuda:
                    mask = mask.cuda()
                for i, length in enumerate(lengths):
                    length = length.item()
                    if (mask[i].size(2) - length) > 0:
                        mask[i].narrow(2, length, mask[i].size(2) - length).fill_(1)
                x = x.masked_fill(mask, 0)
            return x, lengths


    class InferenceBatchSoftmax(nn.Module):
        def forward(self, input_):
            if not self.training:
                return F.softmax(input_, dim=-1)
            else:
                return input_


    class BatchRNN(nn.Module):
        def __init__(self, input_size, hidden_size, rnn_type=nn.LSTM, bidirectional=False, batch_norm=True):
            super(BatchRNN, self).__init__()
            self.input_size = input_size
            self.hidden_size = hidden_size
            self.bidirectional = bidirectional
            self.batch_norm = SequenceWise(nn.BatchNorm1d(input_size)) if batch_norm else None
            self.rnn = rnn_type(input_size=input_size, hidden_size=hidden_size,
                                bidirectional=bidirectional, bias=True)
            self.num_directions = 2 if bidirectional else 1

        def flatten_parameters(self):
            self.rnn.flatten_parameters()

        def forward(self, x, output_lengths):
            if self.batch_norm is not None:
                x = self.batch_norm(x)
            x = nn.utils.rnn.pack_padded_sequence(x, output_lengths, enforce_sorted=False)
            x, h = self.rnn(x)
            x, _ = nn.utils.rnn.pad_packed_sequence(x)
            if self.bidirectional:
                x = x.view(x.size(0), x.size(1), 2, -1).sum(2).view(x.size(0), x.size(1), -1)  # (TxNxH*2) -> (TxNxH) by sum
            return x


    class Lookahead(nn.Module):
        # Wang et al 2016 - Lookahead Convolution Layer for Unidirectional Recurrent Neural Networks
        # input shape - sequence, batch, feature - TxNxH
        # output shape - same as input
        def __init__(self, n_features, context):
            super(Lookahead, self).__init__()
            assert context > 0
            self.context = context
            self.n_features = n_features
            self.pad = (0, self.context - 1)
            self.conv = nn.Conv1d(self.n_features, self.n_features, kernel_size=self.context, stride=1,
                                  groups=self.n_features, padding=0, bias=None)

        def forward(self, x):
            x = x.transpose(0, 1).transpose(1, 2)
            x = F.pad(x, pad=self.pad, value=0)
            x = self.conv(x)
            x = x.transpose(1, 2).transpose(0, 1).contiguous()
            return x

        def __repr__(self):
            return self.__class__.__name__ + '(' \
                   + 'n_features=' + str(self.n_features) \
                   + ', context=' + str(self.context) + ')'

    class DeepSpeech(nn.Module):
        def __init__(self, rnn_type, labels, rnn_hidden_size, nb_layers, audio_conf,
                     bidirectional, context=20):
            super(DeepSpeech, self).__init__()

            self.hidden_size = rnn_hidden_size
            self.hidden_layers = nb_layers
            self.rnn_type = rnn_type
            self.audio_conf = audio_conf
            self.labels = labels
            self.bidirectional = bidirectional

            sample_rate = self.audio_conf["sample_rate"]
            window_size = self.audio_conf["window_size"]
            num_classes = len(self.labels)

            self.conv = MaskConv(nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=(41, 11), stride=(2, 2), padding=(20, 5)),
                nn.BatchNorm2d(32),
                nn.Hardtanh(0, 20, inplace=True),
                nn.Conv2d(32, 32, kernel_size=(21, 11), stride=(2, 1), padding=(10, 5)),
                nn.BatchNorm2d(32),
                nn.Hardtanh(0, 20, inplace=True)
            ))
            # Based on above convolutions and spectrogram size using conv formula (W - F + 2P)/ S+1
            rnn_input_size = int(math.floor((sample_rate * window_size) / 2) + 1)
            rnn_input_size = int(math.floor(rnn_input_size + 2 * 20 - 41) / 2 + 1)
            rnn_input_size = int(math.floor(rnn_input_size + 2 * 10 - 21) / 2 + 1)
            rnn_input_size *= 32

            rnns = []
            rnn = BatchRNN(input_size=rnn_input_size, hidden_size=rnn_hidden_size, rnn_type=rnn_type,
                           bidirectional=bidirectional, batch_norm=False)
            rnns.append(('0', rnn))
            for x in range(nb_layers - 1):
                rnn = BatchRNN(input_size=rnn_hidden_size, hidden_size=rnn_hidden_size, rnn_type=rnn_type,
                               bidirectional=bidirectional)
                rnns.append(('%d' % (x + 1), rnn))
            self.rnns = nn.Sequential(OrderedDict(rnns))
            self.lookahead = nn.Sequential(
                # consider adding batch norm?
                Lookahead(rnn_hidden_size, context=context),
                nn.Hardtanh(0, 20, inplace=True)
            ) if not bidirectional else None

            fully_connected = nn.Sequential(
                nn.BatchNorm1d(rnn_hidden_size),
                nn.Linear(rnn_hidden_size, num_classes, bias=False)
            )
            self.fc = nn.Sequential(
                SequenceWise(fully_connected),
            )
            self.inference_softmax = InferenceBatchSoftmax()

        def forward(self, x, lengths):
            lengths = lengths.cpu().int()
            output_lengths = self.get_seq_lens(lengths)
            x, _ = self.conv(x, output_lengths)

            sizes = x.size()
            x = x.view(sizes[0], sizes[1] * sizes[2], sizes[3])  # Collapse feature dimension
            x = x.transpose(1, 2).transpose(0, 1).contiguous()  # TxNxH

            for rnn in self.rnns:
                x = rnn(x, output_lengths)

            if not self.bidirectional:  # no need for lookahead layer in bidirectional
                x = self.lookahead(x)

            x = self.fc(x)
            x = x.transpose(0, 1)
            # identity in training mode, softmax in eval mode
            x = self.inference_softmax(x)
            return x, output_lengths

        def get_seq_lens(self, input_length):
            """
            Given a 1D Tensor or Variable containing integer sequence lengths, return a 1D tensor or variable
            containing the size sequences that will be output by the network.
            :param input_length: 1D Tensor
            :return: 1D Tensor scaled by model
            """
            seq_len = input_length
            for m in self.conv.modules():
                if type(m) == nn.modules.conv.Conv2d:
                    seq_len = ((seq_len + 2 * m.padding[1] - m.dilation[1] * (m.kernel_size[1] - 1) - 1).true_divide(m.stride[1]) + 1)
            return seq_len.int()


    sample_rate = 16000
    window_size = 0.02
    window_stride = 0.01
    window = "hamming"
    audio_conf = dict(sample_rate=sample_rate,
                      window_size=window_size,
                      window_stride=window_stride,
                      window=window,
                      noise_dir=None)


    N = 10
    num_classes = 10
    spectrogram_size = 161
    # Commented are the original sizes in the code
    seq_length = 500 # 1343
    target_length = 10 # 50
    labels = torch.rand(num_classes, device=device)
    inputs = torch.rand(N, 1, spectrogram_size, seq_length, device=device)
    inputs_sizes = torch.rand(N, device=device).mul(seq_length*0.1).add(seq_length*0.8) # Sequence length for each input
    targets = torch.rand(N, target_length, device=device)
    targets_sizes = torch.full((N,), target_length, dtype=torch.int, device=device)

    model = DeepSpeech(rnn_type=supported_rnns["lstm"], labels=labels, rnn_hidden_size=1024, nb_layers=5,
                       audio_conf=audio_conf, bidirectional=True)
    model = model.to(device)
    criterion = nn.CTCLoss()
    params, names = make_functional(model)

    def forward(*new_params):
        load_weights(model, names, new_params)
        out, out_sizes = model(inputs, inputs_sizes)
        out = out.transpose(0, 1) # For ctc loss

        loss = criterion(out, targets, out_sizes, targets_sizes)
        return loss

    return forward, params

def get_transformer(device):
    # https://github.com/pytorch/examples/blob/master/word_language_model/model.py#L108-L152

    class PositionalEncoding(nn.Module):
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
            self.dropout = nn.Dropout(p=dropout)

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

            x = x + self.pe[:x.size(0), :]
            return self.dropout(x)

    class TransformerModel(nn.Module):
        """Container module with an encoder, a recurrent or transformer module, and a decoder."""

        def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
            super(TransformerModel, self).__init__()
            try:
                from torch.nn import TransformerEncoder, TransformerEncoderLayer
            except:
                raise ImportError('TransformerEncoder module does not exist in PyTorch 1.1 or lower.')
            self.model_type = 'Transformer'
            self.src_mask = None
            self.pos_encoder = PositionalEncoding(ninp, dropout)
            encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
            self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
            self.encoder = nn.Embedding(ntoken, ninp)
            self.ninp = ninp
            self.decoder = nn.Linear(ninp, ntoken)

            self.init_weights()

        def _generate_square_subsequent_mask(self, sz):
            mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
            mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
            return mask

        def init_weights(self):
            initrange = 0.1
            nn.init.uniform_(self.encoder.weight, -initrange, initrange)
            # Not sure how this works in the original code
            # nn.init.zeros_(self.decoder)
            nn.init.uniform_(self.decoder.weight, -initrange, initrange)

        def forward(self, src, has_mask=True):
            if has_mask:
                device = src.device
                if self.src_mask is None or self.src_mask.size(0) != len(src):
                    mask = self._generate_square_subsequent_mask(len(src)).to(device)
                    self.src_mask = mask
            else:
                self.src_mask = None

            src = self.encoder(src) * math.sqrt(self.ninp)
            src = self.pos_encoder(src)
            output = self.transformer_encoder(src, self.src_mask)
            output = self.decoder(output)
            return F.log_softmax(output, dim=-1)

    # For most SOTA research, you would like to have embed to 712, nhead to 12, bsz to 64, tgt_len/src_len to 128.
    N = 64
    seq_length = 128
    ntoken = 50
    model = TransformerModel(ntoken=ntoken, ninp=720, nhead=12, nhid=2048, nlayers=2)
    model.to(device)
    criterion = nn.NLLLoss()
    params, names = make_functional(model)

    data = torch.rand(N, seq_length+1, device=device).mul(ntoken).long()
    inputs = data.narrow(1, 0, seq_length)
    targets = data.narrow(1, 1, seq_length)

    def forward(*new_params):
        load_weights(model, names, new_params)
        out = model(inputs)

        loss = criterion(out.reshape(N * seq_length, ntoken), targets.reshape(N * seq_length))
        return loss

    return forward, params
