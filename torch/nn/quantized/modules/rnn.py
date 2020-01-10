from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numbers

import torch
from torch import nn
from torch.nn import _VF
from torch.nn.utils.rnn import PackedSequence


from torch.nn.rnn import apply_permutation
from torch.nn.quantized.dynamic.rnn import PackedParameter
from torch.nn.quantized.dynamic.rnn import LSTM as DynamicLSTM


class PackedParameter(torch.nn.Module):
    def __init__(self, param):
        """Packed parameter to be used with the statically quantized RNN

        Packed Parameter order:
            (packed_wb_ih, packed_wb_hh,
             linear_ih_scale, linear_ih_zp, linear_hh_scale, linear_hh_zp)

        After calling `unpack()`:
            (w_ih, w_hh, b_ih, b_hh,
             linear_ih_scale, linear_ih_zp,
             linear_hh_scale, linear_hh_zp)
        """
        super(PackedParameter, self).__inti__()
        self.param = param


    @torch.jit.export
    def __getstate__(self):
        w_ih, b_ih = torch.ops.quantized.linear_unpack(self.param[0])
        w_hh, b_hh = torch.ops.quantized.linear_unpack(self.param[1])
        return w_ih, w_hh, b_ih, b_hh, self.param[2:]

    @torch.jit.export
    def __setstate(self, state):
        wb_ih = torch.ops.quantized.linear_prepack(state[0], state[2])
        wb_hh = torch.ops.quantized.linear_prepack(state[1], state[3])
        self.param = (wb_ih, )

class LSTM(DynamicLSTM):
    r"""Quantized LSTM (static).

    Linear output range: -8, 8
    Linear output scale: 16 / 256
    Linear output zp: 0
    """
    def _get_name(self):
      return 'StaticQuantizedLSTM'

    def forward_impl(self, input, hx, batch_sizes, max_batch_size, sorted_indices):
        # type: (Tensor, Optional[Tuple[Tensor, Tensor]], Optional[Tensor], int, Optional[Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]  # noqa

       if hx is None:
              num_directions = 2 if self.bidirectional else 1
              zeros = torch.zeros(self.num_layers * num_directions,
                                  max_batch_size, self.hidden_size,
                                  dtype=input.dtype, device=input.device)
              hx = (zeros, zeros)
          else:
              # Each batch of the hidden state should match the input sequence that
              # the user believes he/she is passing in.
              hx = self.permute_hidden(hx, sorted_indices)

          self.check_forward_args(input, hx, batch_sizes)

          weight_values = []
          for mod in self._all_weight_values:
              weight_values.append(mod.param)

          if batch_sizes is None:
              result = _VF.quantized_lstm(input, hx, weight_values, self.bias, self.num_layers,
                                          float(self.dropout), self.training, self.bidirectional,
                                          self.batch_first, dtype=self.dtype, use_dynamic=False)
          else:
              result = _VF.quantized_lstm(input, batch_sizes, hx, weight_values, self.bias,
                                          self.num_layers, float(self.dropout), self.training,
                                          self.bidirectional, dtype=self.dtype, use_dynamic=False)
          output = result[0]
          hidden = result[1:]
          return output, hidden
