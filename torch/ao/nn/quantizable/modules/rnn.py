"""
We will recreate all the RNN modules as we require the modules to be decomposed
into its building blocks to be able to observe.
"""

# mypy: allow-untyped-defs

import numbers
import warnings
from typing import Optional

import torch
from torch import Tensor


__all__ = ["LSTMCell", "LSTM"]


class LSTMCell(torch.nn.Module):
    r"""A quantizable long short-term memory (LSTM) cell.

    For the description and the argument types, please, refer to :class:`~torch.nn.LSTMCell`

    `split_gates`: specify True to compute the input/forget/cell/output gates separately
    to avoid an intermediate tensor which is subsequently chunk'd. This optimization can
    be beneficial for on-device inference latency. This flag is cascaded down from the
    parent classes.

    Examples::

        >>> import torch.ao.nn.quantizable as nnqa
        >>> rnn = nnqa.LSTMCell(10, 20)
        >>> input = torch.randn(6, 10)
        >>> hx = torch.randn(3, 20)
        >>> cx = torch.randn(3, 20)
        >>> output = []
        >>> for i in range(6):
        ...     hx, cx = rnn(input[i], (hx, cx))
        ...     output.append(hx)
    """

    _FLOAT_MODULE = torch.nn.LSTMCell
    __constants__ = ["split_gates"]  # for jit.script

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        bias: bool = True,
        device=None,
        dtype=None,
        *,
        split_gates=False,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.input_size = input_dim
        self.hidden_size = hidden_dim
        self.bias = bias
        self.split_gates = split_gates

        if not split_gates:
            self.igates: torch.nn.Module = torch.nn.Linear(
                input_dim, 4 * hidden_dim, bias=bias, **factory_kwargs
            )
            self.hgates: torch.nn.Module = torch.nn.Linear(
                hidden_dim, 4 * hidden_dim, bias=bias, **factory_kwargs
            )
            self.gates: torch.nn.Module = torch.ao.nn.quantized.FloatFunctional()
        else:
            # keep separate Linear layers for each gate
            self.igates = torch.nn.ModuleDict()
            self.hgates = torch.nn.ModuleDict()
            self.gates = torch.nn.ModuleDict()
            for g in ["input", "forget", "cell", "output"]:
                # pyre-fixme[29]: `Union[torch._tensor.Tensor, torch.nn.modules.module.Module]`
                self.igates[g] = torch.nn.Linear(
                    input_dim, hidden_dim, bias=bias, **factory_kwargs
                )
                # pyre-fixme[29]: `Union[torch._tensor.Tensor, torch.nn.modules.module.Module]`
                self.hgates[g] = torch.nn.Linear(
                    hidden_dim, hidden_dim, bias=bias, **factory_kwargs
                )
                # pyre-fixme[29]: `Union[torch._tensor.Tensor, torch.nn.modules.module.Module]`
                self.gates[g] = torch.ao.nn.quantized.FloatFunctional()

        self.input_gate = torch.nn.Sigmoid()
        self.forget_gate = torch.nn.Sigmoid()
        self.cell_gate = torch.nn.Tanh()
        self.output_gate = torch.nn.Sigmoid()

        self.fgate_cx = torch.ao.nn.quantized.FloatFunctional()
        self.igate_cgate = torch.ao.nn.quantized.FloatFunctional()
        self.fgate_cx_igate_cgate = torch.ao.nn.quantized.FloatFunctional()

        self.ogate_cy = torch.ao.nn.quantized.FloatFunctional()

        self.initial_hidden_state_qparams: tuple[float, int] = (1.0, 0)
        self.initial_cell_state_qparams: tuple[float, int] = (1.0, 0)
        self.hidden_state_dtype: torch.dtype = torch.quint8
        self.cell_state_dtype: torch.dtype = torch.quint8

    def forward(
        self, x: Tensor, hidden: Optional[tuple[Tensor, Tensor]] = None
    ) -> tuple[Tensor, Tensor]:
        if hidden is None or hidden[0] is None or hidden[1] is None:
            hidden = self.initialize_hidden(x.shape[0], x.is_quantized)
        hx, cx = hidden

        if not self.split_gates:
            igates = self.igates(x)
            hgates = self.hgates(hx)
            gates = self.gates.add(igates, hgates)  # type: ignore[operator]

            input_gate, forget_gate, cell_gate, out_gate = gates.chunk(4, 1)

            input_gate = self.input_gate(input_gate)
            forget_gate = self.forget_gate(forget_gate)
            cell_gate = self.cell_gate(cell_gate)
            out_gate = self.output_gate(out_gate)
        else:
            # apply each input + hidden projection and add together
            gate = {}
            for (key, gates), igates, hgates in zip(
                self.gates.items(),  # type: ignore[operator]
                self.igates.values(),  # type: ignore[operator]
                self.hgates.values(),  # type: ignore[operator]
            ):
                gate[key] = gates.add(igates(x), hgates(hx))

            input_gate = self.input_gate(gate["input"])
            forget_gate = self.forget_gate(gate["forget"])
            cell_gate = self.cell_gate(gate["cell"])
            out_gate = self.output_gate(gate["output"])

        fgate_cx = self.fgate_cx.mul(forget_gate, cx)
        igate_cgate = self.igate_cgate.mul(input_gate, cell_gate)
        fgate_cx_igate_cgate = self.fgate_cx_igate_cgate.add(fgate_cx, igate_cgate)
        cy = fgate_cx_igate_cgate

        # TODO: make this tanh a member of the module so its qparams can be configured
        tanh_cy = torch.tanh(cy)
        hy = self.ogate_cy.mul(out_gate, tanh_cy)
        return hy, cy

    def initialize_hidden(
        self, batch_size: int, is_quantized: bool = False
    ) -> tuple[Tensor, Tensor]:
        h, c = (
            torch.zeros((batch_size, self.hidden_size)),
            torch.zeros((batch_size, self.hidden_size)),
        )
        if is_quantized:
            (h_scale, h_zp) = self.initial_hidden_state_qparams
            (c_scale, c_zp) = self.initial_cell_state_qparams
            h = torch.quantize_per_tensor(
                h, scale=h_scale, zero_point=h_zp, dtype=self.hidden_state_dtype
            )
            c = torch.quantize_per_tensor(
                c, scale=c_scale, zero_point=c_zp, dtype=self.cell_state_dtype
            )
        return h, c

    def _get_name(self):
        return "QuantizableLSTMCell"

    @classmethod
    def from_params(cls, wi, wh, bi=None, bh=None, split_gates=False):
        """Uses the weights and biases to create a new LSTM cell.

        Args:
            wi, wh: Weights for the input and hidden layers
            bi, bh: Biases for the input and hidden layers
        """
        assert (bi is None) == (bh is None)  # Either both None or both have values
        input_size = wi.shape[1]
        hidden_size = wh.shape[1]
        cell = cls(
            input_dim=input_size,
            hidden_dim=hidden_size,
            bias=(bi is not None),
            split_gates=split_gates,
        )

        if not split_gates:
            cell.igates.weight = torch.nn.Parameter(wi)
            if bi is not None:
                cell.igates.bias = torch.nn.Parameter(bi)
            cell.hgates.weight = torch.nn.Parameter(wh)
            if bh is not None:
                cell.hgates.bias = torch.nn.Parameter(bh)
        else:
            # split weight/bias
            for w, b, gates in zip([wi, wh], [bi, bh], [cell.igates, cell.hgates]):
                for w_chunk, gate in zip(w.chunk(4, dim=0), gates.values()):  # type: ignore[operator]
                    gate.weight = torch.nn.Parameter(w_chunk)

                if b is not None:
                    for b_chunk, gate in zip(b.chunk(4, dim=0), gates.values()):  # type: ignore[operator]
                        gate.bias = torch.nn.Parameter(b_chunk)

        return cell

    @classmethod
    def from_float(cls, other, use_precomputed_fake_quant=False, split_gates=False):
        assert type(other) == cls._FLOAT_MODULE
        assert hasattr(other, "qconfig"), "The float module must have 'qconfig'"
        observed = cls.from_params(
            other.weight_ih,
            other.weight_hh,
            other.bias_ih,
            other.bias_hh,
            split_gates=split_gates,
        )
        observed.qconfig = other.qconfig
        observed.igates.qconfig = other.qconfig
        observed.hgates.qconfig = other.qconfig
        if split_gates:
            # also apply qconfig directly to Linear modules
            for g in observed.igates.values():
                g.qconfig = other.qconfig
            for g in observed.hgates.values():
                g.qconfig = other.qconfig
        return observed


class _LSTMSingleLayer(torch.nn.Module):
    r"""A single one-directional LSTM layer.

    The difference between a layer and a cell is that the layer can process a
    sequence, while the cell only expects an instantaneous value.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        bias: bool = True,
        device=None,
        dtype=None,
        *,
        split_gates=False,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.cell = LSTMCell(
            input_dim, hidden_dim, bias=bias, split_gates=split_gates, **factory_kwargs
        )

    def forward(self, x: Tensor, hidden: Optional[tuple[Tensor, Tensor]] = None):
        result = []
        seq_len = x.shape[0]
        for i in range(seq_len):
            hidden = self.cell(x[i], hidden)
            result.append(hidden[0])  # type: ignore[index]
        result_tensor = torch.stack(result, 0)
        return result_tensor, hidden

    @classmethod
    def from_params(cls, *args, **kwargs):
        cell = LSTMCell.from_params(*args, **kwargs)
        layer = cls(
            cell.input_size, cell.hidden_size, cell.bias, split_gates=cell.split_gates
        )
        layer.cell = cell
        return layer


class _LSTMLayer(torch.nn.Module):
    r"""A single bi-directional LSTM layer."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        bias: bool = True,
        batch_first: bool = False,
        bidirectional: bool = False,
        device=None,
        dtype=None,
        *,
        split_gates=False,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.batch_first = batch_first
        self.bidirectional = bidirectional
        self.layer_fw = _LSTMSingleLayer(
            input_dim, hidden_dim, bias=bias, split_gates=split_gates, **factory_kwargs
        )
        if self.bidirectional:
            self.layer_bw = _LSTMSingleLayer(
                input_dim,
                hidden_dim,
                bias=bias,
                split_gates=split_gates,
                **factory_kwargs,
            )

    def forward(self, x: Tensor, hidden: Optional[tuple[Tensor, Tensor]] = None):
        if self.batch_first:
            x = x.transpose(0, 1)
        if hidden is None:
            hx_fw, cx_fw = (None, None)
        else:
            hx_fw, cx_fw = hidden
        hidden_bw: Optional[tuple[Tensor, Tensor]] = None
        if self.bidirectional:
            if hx_fw is None:
                hx_bw = None
            else:
                hx_bw = hx_fw[1]
                hx_fw = hx_fw[0]
            if cx_fw is None:
                cx_bw = None
            else:
                cx_bw = cx_fw[1]
                cx_fw = cx_fw[0]
            if hx_bw is not None and cx_bw is not None:
                hidden_bw = hx_bw, cx_bw
        if hx_fw is None and cx_fw is None:
            hidden_fw = None
        else:
            hidden_fw = (
                torch.jit._unwrap_optional(hx_fw),
                torch.jit._unwrap_optional(cx_fw),
            )
        result_fw, hidden_fw = self.layer_fw(x, hidden_fw)

        if hasattr(self, "layer_bw") and self.bidirectional:
            x_reversed = x.flip(0)
            result_bw, hidden_bw = self.layer_bw(x_reversed, hidden_bw)
            result_bw = result_bw.flip(0)

            result = torch.cat([result_fw, result_bw], result_fw.dim() - 1)
            if hidden_fw is None and hidden_bw is None:
                h = None
                c = None
            elif hidden_fw is None:
                (h, c) = torch.jit._unwrap_optional(hidden_bw)
            elif hidden_bw is None:
                (h, c) = torch.jit._unwrap_optional(hidden_fw)
            else:
                h = torch.stack([hidden_fw[0], hidden_bw[0]], 0)  # type: ignore[list-item]
                c = torch.stack([hidden_fw[1], hidden_bw[1]], 0)  # type: ignore[list-item]
        else:
            result = result_fw
            h, c = torch.jit._unwrap_optional(hidden_fw)  # type: ignore[assignment]

        if self.batch_first:
            result.transpose_(0, 1)

        return result, (h, c)

    @classmethod
    def from_float(cls, other, layer_idx=0, qconfig=None, **kwargs):
        r"""
        There is no FP equivalent of this class. This function is here just to
        mimic the behavior of the `prepare` within the `torch.ao.quantization`
        flow.
        """
        assert hasattr(other, "qconfig") or (qconfig is not None)

        input_size = kwargs.get("input_size", other.input_size)
        hidden_size = kwargs.get("hidden_size", other.hidden_size)
        bias = kwargs.get("bias", other.bias)
        batch_first = kwargs.get("batch_first", other.batch_first)
        bidirectional = kwargs.get("bidirectional", other.bidirectional)
        split_gates = kwargs.get("split_gates", False)

        layer = cls(
            input_size,
            hidden_size,
            bias,
            batch_first,
            bidirectional,
            split_gates=split_gates,
        )
        # pyrefly: ignore  # bad-argument-type
        layer.qconfig = getattr(other, "qconfig", qconfig)
        wi = getattr(other, f"weight_ih_l{layer_idx}")
        wh = getattr(other, f"weight_hh_l{layer_idx}")
        bi = getattr(other, f"bias_ih_l{layer_idx}", None)
        bh = getattr(other, f"bias_hh_l{layer_idx}", None)

        layer.layer_fw = _LSTMSingleLayer.from_params(
            wi, wh, bi, bh, split_gates=split_gates
        )

        if other.bidirectional:
            wi = getattr(other, f"weight_ih_l{layer_idx}_reverse")
            wh = getattr(other, f"weight_hh_l{layer_idx}_reverse")
            bi = getattr(other, f"bias_ih_l{layer_idx}_reverse", None)
            bh = getattr(other, f"bias_hh_l{layer_idx}_reverse", None)
            layer.layer_bw = _LSTMSingleLayer.from_params(
                wi, wh, bi, bh, split_gates=split_gates
            )
        return layer


class LSTM(torch.nn.Module):
    r"""A quantizable long short-term memory (LSTM).

    For the description and the argument types, please, refer to :class:`~torch.nn.LSTM`

    Attributes:
        layers : instances of the `_LSTMLayer`

    .. note::
        To access the weights and biases, you need to access them per layer.
        See examples below.

    Examples::

        >>> import torch.ao.nn.quantizable as nnqa
        >>> rnn = nnqa.LSTM(10, 20, 2)
        >>> input = torch.randn(5, 3, 10)
        >>> h0 = torch.randn(2, 3, 20)
        >>> c0 = torch.randn(2, 3, 20)
        >>> output, (hn, cn) = rnn(input, (h0, c0))
        >>> # To get the weights:
        >>> # xdoctest: +SKIP
        >>> print(rnn.layers[0].weight_ih)
        tensor([[...]])
        >>> print(rnn.layers[0].weight_hh)
        AssertionError: There is no reverse path in the non-bidirectional layer
    """

    _FLOAT_MODULE = torch.nn.LSTM

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bias: bool = True,
        batch_first: bool = False,
        dropout: float = 0.0,
        bidirectional: bool = False,
        device=None,
        dtype=None,
        *,
        split_gates: bool = False,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = float(dropout)
        self.bidirectional = bidirectional
        self.training = False  # Default to eval mode. If we want to train, we will explicitly set to training.

        if (
            not isinstance(dropout, numbers.Number)
            # pyrefly: ignore  # unsupported-operation
            or not 0 <= dropout <= 1
            or isinstance(dropout, bool)
        ):
            raise ValueError(
                "dropout should be a number in range [0, 1] "
                "representing the probability of an element being "
                "zeroed"
            )
        # pyrefly: ignore  # unsupported-operation
        if dropout > 0:
            warnings.warn(
                "dropout option for quantizable LSTM is ignored. "
                "If you are training, please, use nn.LSTM version "
                "followed by `prepare` step."
            )
            if num_layers == 1:
                warnings.warn(
                    "dropout option adds dropout after all but last "
                    "recurrent layer, so non-zero dropout expects "
                    f"num_layers greater than 1, but got dropout={dropout} "
                    f"and num_layers={num_layers}"
                )

        layers = [
            _LSTMLayer(
                self.input_size,
                self.hidden_size,
                self.bias,
                batch_first=False,
                bidirectional=self.bidirectional,
                split_gates=split_gates,
                **factory_kwargs,
            )
        ]
        layers.extend(
            _LSTMLayer(
                self.hidden_size,
                self.hidden_size,
                self.bias,
                batch_first=False,
                bidirectional=self.bidirectional,
                split_gates=split_gates,
                **factory_kwargs,
            )
            for _ in range(1, num_layers)
        )
        self.layers = torch.nn.ModuleList(layers)

    def forward(self, x: Tensor, hidden: Optional[tuple[Tensor, Tensor]] = None):
        if self.batch_first:
            x = x.transpose(0, 1)

        max_batch_size = x.size(1)
        num_directions = 2 if self.bidirectional else 1
        if hidden is None:
            zeros = torch.zeros(
                num_directions,
                max_batch_size,
                self.hidden_size,
                dtype=torch.float,
                device=x.device,
            )
            zeros.squeeze_(0)
            if x.is_quantized:
                zeros = torch.quantize_per_tensor(
                    zeros, scale=1.0, zero_point=0, dtype=x.dtype
                )
            hxcx = [(zeros, zeros) for _ in range(self.num_layers)]
        else:
            hidden_non_opt = torch.jit._unwrap_optional(hidden)
            if isinstance(hidden_non_opt[0], Tensor):
                hx = hidden_non_opt[0].reshape(
                    self.num_layers, num_directions, max_batch_size, self.hidden_size
                )
                cx = hidden_non_opt[1].reshape(
                    self.num_layers, num_directions, max_batch_size, self.hidden_size
                )
                hxcx = [
                    (hx[idx].squeeze(0), cx[idx].squeeze(0))
                    for idx in range(self.num_layers)
                ]
            else:
                hxcx = hidden_non_opt

        hx_list = []
        cx_list = []
        for idx, layer in enumerate(self.layers):
            x, (h, c) = layer(x, hxcx[idx])
            hx_list.append(torch.jit._unwrap_optional(h))
            cx_list.append(torch.jit._unwrap_optional(c))
        hx_tensor = torch.stack(hx_list)
        cx_tensor = torch.stack(cx_list)

        # We are creating another dimension for bidirectional case
        # need to collapse it
        hx_tensor = hx_tensor.reshape(-1, hx_tensor.shape[-2], hx_tensor.shape[-1])
        cx_tensor = cx_tensor.reshape(-1, cx_tensor.shape[-2], cx_tensor.shape[-1])

        if self.batch_first:
            x = x.transpose(0, 1)

        return x, (hx_tensor, cx_tensor)

    def _get_name(self):
        return "QuantizableLSTM"

    @classmethod
    def from_float(cls, other, qconfig=None, split_gates=False):
        assert isinstance(other, cls._FLOAT_MODULE)
        assert hasattr(other, "qconfig") or qconfig
        observed = cls(
            other.input_size,
            other.hidden_size,
            other.num_layers,
            other.bias,
            other.batch_first,
            other.dropout,
            other.bidirectional,
            split_gates=split_gates,
        )
        # pyrefly: ignore  # bad-argument-type
        observed.qconfig = getattr(other, "qconfig", qconfig)
        for idx in range(other.num_layers):
            observed.layers[idx] = _LSTMLayer.from_float(
                other, idx, qconfig, batch_first=False, split_gates=split_gates
            )

        # Prepare the model
        if other.training:
            observed.train()
            observed = torch.ao.quantization.prepare_qat(observed, inplace=True)
        else:
            observed.eval()
            observed = torch.ao.quantization.prepare(observed, inplace=True)
        return observed

    @classmethod
    def from_observed(cls, other):
        # The whole flow is float -> observed -> quantized
        # This class does float -> observed only
        raise NotImplementedError(
            "It looks like you are trying to convert a "
            "non-quantizable LSTM module. Please, see "
            "the examples on quantizable LSTMs."
        )
