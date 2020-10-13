import numbers
from typing import Optional, Tuple

import torch
from torch import Tensor
import torch.nn as nn

from .functional_modules import FloatFunctional


"""
We will recreate all the RNN modules as we require the modules to be decomposed
into its building blocks to be able to observe.
"""

def _hidden_as_output(x):
    if isinstance(x, Tensor):
        return x
    return x[0]

class LSTMCell(torch.nn.Module):
    _FLOAT_MODULE = nn.LSTMCell

    def __init__(self, input_dim, hidden_dim, bias=True):
        super().__init__()
        self.input_size = input_dim
        self.hidden_size = hidden_dim
        self.bias = bias

        self.igates = nn.Linear(input_dim, 4 * hidden_dim, bias=bias)
        self.hgates = nn.Linear(hidden_dim, 4 * hidden_dim, bias=bias)
        self.gates = FloatFunctional()

        self.fgate_cx = FloatFunctional()
        self.igate_cgate = FloatFunctional()
        self.fgate_cx_igate_cgate = FloatFunctional()

        self.ogate_cy = FloatFunctional()

    def forward(self, x: Tensor, hidden: Optional[Tuple[Tensor, Tensor]] = None) -> Tuple[Tensor, Tensor]:
        if hidden is None or hidden == (None, None):
            hidden = self.initialize_hidden(x.shape[0], x.is_quantized)
        hx, cx = hidden

        igates = self.igates(x)
        hgates = self.hgates(hx)
        gates = self.gates.add(igates, hgates)

        input_gate, forget_gate, cell_gate, out_gate = gates.chunk(4, 1)

        input_gate = torch.sigmoid(input_gate)
        forget_gate = torch.sigmoid(forget_gate)
        cell_gate = torch.tanh(cell_gate)
        out_gate = torch.sigmoid(out_gate)

        fgate_cx = self.fgate_cx.mul(forget_gate, cx)
        igate_cgate = self.igate_cgate.mul(input_gate, cell_gate)
        fgate_cx_igate_cgate = self.fgate_cx_igate_cgate.add(fgate_cx, igate_cgate)
        cy = fgate_cx_igate_cgate

        tanh_cy = torch.tanh(cy)
        hy = self.ogate_cy.mul(out_gate, tanh_cy)

        return hy, (hy, cy)

    def initialize_hidden(self, batch_size: int, is_quantized: bool = False) -> Tuple[Tensor, Tensor]:
        h, c = torch.zeros((batch_size, self.hidden_size)), torch.zeros((batch_size, self.hidden_size))
        if is_quantized:
            h = torch.quantize_per_tensor(h, scale=1.0, zero_point=0, dtype=torch.quint8)
            c = torch.quantize_per_tensor(c, scale=1.0, zero_point=0, dtype=torch.quint8)
        return h, c

    @classmethod
    def from_params(cls, wi, wh, bi=None, bh=None):
        assert (bi is None) == (bh is None)  # Either both None or both have values
        input_size = wi.shape[1]
        hidden_size = wh.shape[1]
        cell = cls(input_dim=input_size, hidden_dim=hidden_size,
                   bias=(bi is not None))
        cell.igates.weight = nn.Parameter(wi)
        if bi is not None:
            cell.igates.bias = nn.Parameter(bi)
        cell.hgates.weight = nn.Parameter(wh)
        if bh is not None:
            cell.hgates.bias = nn.Parameter(bh)
        return cell

    @classmethod
    def from_float(cls, other):
        assert hasattr(other, 'qconfig'), "The float module must have 'qconfig'"
        observed = cls.from_params(other.weight_ih, other.weight_hh,
                                   other.bias_ih, other.bias_hh)
        observed.qconfig = other.qconfig
        observed.igates.qconfig = other.qconfig
        observed.hgates.qconfig = other.qconfig
        return observed


class _LSTMLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, bias=True):
        super().__init__()
        self.cell = LSTMCell(input_dim, hidden_dim, bias=bias)

    def forward(self, x, hidden=None):
        result = []
        for xx in x:
            output, hidden = self.cell(xx, hidden)
            result.append(_hidden_as_output(hidden))
        result = torch.stack(result, 0)
        return result, hidden

    @classmethod
    def from_params(cls, *args, **kwargs):
        cell = LSTMCell.from_params(*args, **kwargs)
        layer = cls(cell.input_size, cell.hidden_size, cell.bias)
        layer.cell = cell
        return layer


class LSTMLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, bias=True, batch_first=False,
                 bidirectional=False):
        super().__init__()
        self.batch_first = batch_first
        self.bidirectional = bidirectional
        self.layer_fw = _LSTMLayer(input_dim, hidden_dim, bias=bias)
        if self.bidirectional:
            self.layer_bw = _LSTMLayer(input_dim, hidden_dim, bias=bias)

    def forward(self, x, hidden=None):
        if self.batch_first:
            x = x.transpose(0, 1)
        if hidden is None:
            hidden = (None, None)
        hx_fw, cx_fw = hidden
        if self.bidirectional:
            if hx_fw is None:
                hx_fw = [None, None]
            if cx_fw is None:
                cx_fw = [None, None]
            hx_bw = hx_fw[1]
            hx_fw = hx_fw[0]
            cx_bw = cx_fw[1]
            cx_fw = cx_fw[0]
            hidden_bw = hx_bw, cx_bw
        hidden_fw = hx_fw, cx_fw
        result_fw, hidden_fw = self.layer_fw(x, hidden_fw)
        if not self.bidirectional:
            return result_fw, hidden_fw

        x_reversed = x.flip(0)
        result_bw, hidden_bw = self.layer_bw(x_reversed, hidden_bw)
        result_bw = result_bw.flip(0)

        result = torch.cat([result_fw, result_bw], result_fw.dim() - 1)
        h = torch.stack([hidden_fw[0], hidden_bw[0]], 0)
        c = torch.stack([hidden_fw[1], hidden_bw[1]], 0)
        if self.batch_first:
            result.transpose_(0, 1)
        return result, (h, c)

    @classmethod
    def from_float(cls, other, layer_idx=0, qconfig=None, **kwargs):
        assert hasattr(other, 'qconfig') or (qconfig is not None)

        input_size = kwargs.get('input_size', other.input_size)
        hidden_size = kwargs.get('hidden_size', other.hidden_size)
        bias = kwargs.get('bias', other.bias)
        batch_first = kwargs.get('batch_first', other.batch_first)
        bidirectional = kwargs.get('bidirectional', other.bidirectional)

        layer = cls(input_size, hidden_size, bias, batch_first, bidirectional)
        layer.qconfig = getattr(other, 'qconfig', qconfig)
        wi = getattr(other, f'weight_ih_l{layer_idx}')
        wh = getattr(other, f'weight_hh_l{layer_idx}')
        bi = getattr(other, f'bias_ih_l{layer_idx}', None)
        bh = getattr(other, f'bias_hh_l{layer_idx}', None)

        layer.layer_fw = _LSTMLayer.from_params(wi, wh, bi, bh)

        if other.bidirectional:
            wi = getattr(other, f'weight_ih_l{layer_idx}_reverse')
            wh = getattr(other, f'weight_hh_l{layer_idx}_reverse')
            bi = getattr(other, f'bias_ih_l{layer_idx}_reverse', None)
            bh = getattr(other, f'bias_hh_l{layer_idx}_reverse', None)
            layer.layer_bw = _LSTMLayer.from_params(wi, wh, bi, bh)
        return layer


class LSTM(nn.Module):
    _FLOAT_MODULE = nn.LSTM

    def __init__(self, input_size: int, hidden_size: int,
                 num_layers: int = 1, bias: bool = True, batch_first: bool = False,
                 dropout: float = 0., bidirectional: bool = False):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = float(dropout)
        self.bidirectional = bidirectional
        self.training = False  # We don't want to train using this module
        num_directions = 2 if bidirectional else 1

        if not isinstance(dropout, numbers.Number) or not 0 <= dropout <= 1 or \
                isinstance(dropout, bool):
            raise ValueError("dropout should be a number in range [0, 1] "
                             "representing the probability of an element being "
                             "zeroed")
        if dropout > 0:
            warnings.warn("dropout option for quantizable LSTM is ignored. "
                          "If you are training, please, use  nn.LSTM version "
                          "followed by `prepare` step.")
            if num_layers == 1:
                warnings.warn("dropout option adds dropout after all but last "
                              "recurrent layer, so non-zero dropout expects "
                              "num_layers greater than 1, but got dropout={} "
                              "and num_layers={}".format(dropout, num_layers))

        layers = [LSTMLayer(self.input_size, self.hidden_size,
                            self.bias, batch_first=False,
                            bidirectional=self.bidirectional)]
        for layer in range(1, num_layers):
            layers.append(LSTMLayer(self.hidden_size, self.hidden_size,
                                    self.bias, batch_first=False,
                                    bidirectional=self.bidirectional))
        self.layers = nn.ModuleList(layers)

    def forward(self, x, hidden=None):
        if self.batch_first:
            x = x.transpose(0, 1)

        max_batch_size = x.size(1)
        num_directions = 2 if self.bidirectional else 1
        if hidden is None:
            zeros = torch.zeros(num_directions, max_batch_size,
                                self.hidden_size, dtype=torch.float,
                                device=x.device)
            zeros.squeeze_(0)
            if x.is_quantized:
                zeros = torch.quantize_per_tensor(zeros, scale=1.0,
                                                  zero_point=0, dtype=x.dtype)
            hidden = [(zeros, zeros) for _ in range(self.num_layers)]
        elif isinstance(hidden[0], Tensor):
            hx = hidden[0].reshape(self.num_layers, num_directions,
                                   max_batch_size, self.hidden_size).unbind(0)
            cx = hidden[1].reshape(self.num_layers, num_directions,
                                   max_batch_size, self.hidden_size).unbind(0)
            hidden = []
            for idx in range(self.num_layers):
                hidden.append((hx[idx].squeeze_(0), cx[idx].squeeze_(0)))

        for idx in range(self.num_layers):
            x, hidden[idx] = self.layers[idx](x, hidden[idx])

        hx = []
        cx = []
        for idx in range(self.num_layers):
            hx.append(hidden[idx][0])
            cx.append(hidden[idx][1])
        hx = torch.stack(hx)
        cx = torch.stack(cx)

        # We are creating another dimension for bidirectional case
        # need to collapse it
        hx = hx.reshape(-1, *hx.shape[-2:])
        cx = cx.reshape(-1, *hx.shape[-2:])

        if self.batch_first:
            x = x.transpose(0, 1)

        return x, (hx, cx)

    @classmethod
    def from_float(cls, other, qconfig=None):
        assert isinstance(other, cls._FLOAT_MODULE)
        assert (hasattr(other, 'qconfig') or qconfig)
        observed = cls(other.input_size, other.hidden_size, other.num_layers,
                       other.bias, other.batch_first, other.dropout,
                       other.bidirectional)
        observed.qconfig = getattr(other, 'qconfig', qconfig)
        for idx in range(other.num_layers):
            observed.layers[idx] = LSTMLayer.from_float(other, idx, qconfig,
                                                        batch_first=False)
        observed.eval()
        return observed
