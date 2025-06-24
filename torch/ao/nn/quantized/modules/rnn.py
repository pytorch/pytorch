# mypy: allow-untyped-defs
import torch


__all__ = [
    "LSTM",
]


class LSTM(torch.ao.nn.quantizable.LSTM):
    r"""A quantized long short-term memory (LSTM).

    For the description and the argument types, please, refer to :class:`~torch.nn.LSTM`

    Attributes:
        layers : instances of the `_LSTMLayer`

    .. note::
        To access the weights and biases, you need to access them per layer.
        See examples in :class:`~torch.ao.nn.quantizable.LSTM`

    Examples::
        >>> # xdoctest: +SKIP
        >>> custom_module_config = {
        ...     'float_to_observed_custom_module_class': {
        ...         nn.LSTM: nn.quantizable.LSTM,
        ...     },
        ...     'observed_to_quantized_custom_module_class': {
        ...         nn.quantizable.LSTM: nn.quantized.LSTM,
        ...     }
        ... }
        >>> tq.prepare(model, prepare_custom_module_class=custom_module_config)
        >>> tq.convert(model, convert_custom_module_class=custom_module_config)
    """

    _FLOAT_MODULE = torch.ao.nn.quantizable.LSTM  # type: ignore[assignment]

    def _get_name(self):
        return "QuantizedLSTM"

    @classmethod
    def from_float(cls, *args, **kwargs):
        # The whole flow is float -> observed -> quantized
        # This class does observed -> quantized only
        raise NotImplementedError(
            "It looks like you are trying to convert a "
            "non-observed LSTM module. Please, see "
            "the examples on quantizable LSTMs."
        )

    @classmethod
    def from_observed(cls, other):
        assert isinstance(other, cls._FLOAT_MODULE)  # type: ignore[has-type]
        converted = torch.ao.quantization.convert(
            other, inplace=False, remove_qconfig=True
        )
        converted.__class__ = cls
        return converted
