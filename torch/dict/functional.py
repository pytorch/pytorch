from __future__ import annotations

from typing import Sequence, TypeVar

import torch
from .base import _is_tensor_collection
from .tensordict import TensorDict
from .utils import _check_keys, _shape, DeviceType

T = TypeVar("T", bound="TensorDictBase")


def pad(tensordict: T, pad_size: Sequence[int], value: float = 0.0) -> T:
    """Pads all tensors in a tensordict along the batch dimensions with a constant value, returning a new tensordict.

    Args:
         tensordict (TensorDict): The tensordict to pad
         pad_size (Sequence[int]): The padding size by which to pad some batch
            dimensions of the tensordict, starting from the first dimension and
            moving forward. [len(pad_size) / 2] dimensions of the batch size will
            be padded. For example to pad only the first dimension, pad has the form
            (padding_left, padding_right). To pad two dimensions,
            (padding_left, padding_right, padding_top, padding_bottom) and so on.
            pad_size must be even and less than or equal to twice the number of batch dimensions.
         value (float, optional): The fill value to pad by, default 0.0

    Returns:
        A new TensorDict padded along the batch dimensions

    Examples:
        >>> from torch.dict import TensorDict, pad
        >>> import torch
        >>> td = TensorDict({'a': torch.ones(3, 4, 1),
        ...     'b': torch.ones(3, 4, 1, 1)}, batch_size=[3, 4])
        >>> dim0_left, dim0_right, dim1_left, dim1_right = [0, 1, 0, 2]
        >>> padded_td = pad(td, [dim0_left, dim0_right, dim1_left, dim1_right], value=0.0)
        >>> print(padded_td.batch_size)
        torch.Size([4, 6])
        >>> print(padded_td.get("a").shape)
        torch.Size([4, 6, 1])
        >>> print(padded_td.get("b").shape)
        torch.Size([4, 6, 1, 1])

    """
    if len(pad_size) > 2 * len(tensordict.batch_size):
        raise RuntimeError(
            "The length of pad_size must be <= 2 * the number of batch dimensions"
        )

    if len(pad_size) % 2:
        raise RuntimeError("pad_size must have an even number of dimensions")

    new_batch_size = list(tensordict.batch_size)
    for i in range(len(pad_size)):
        new_batch_size[i // 2] += pad_size[i]

    reverse_pad = pad_size[::-1]
    for i in range(0, len(reverse_pad), 2):
        reverse_pad[i], reverse_pad[i + 1] = reverse_pad[i + 1], reverse_pad[i]

    out = TensorDict(
        {}, torch.Size(new_batch_size), device=tensordict.device, _run_checks=False
    )
    for key, tensor in tensordict.items():
        cur_pad = reverse_pad
        if len(pad_size) < len(_shape(tensor)) * 2:
            cur_pad = [0] * (len(_shape(tensor)) * 2 - len(pad_size)) + reverse_pad

        if _is_tensor_collection(tensor.__class__):
            padded = pad(tensor, pad_size, value)
        else:
            padded = torch.nn.functional.pad(tensor, cur_pad, value=value)
        out.set(key, padded)

    return out


def pad_sequence(
    list_of_tensordicts: Sequence[T],
    batch_first: bool = True,
    padding_value: float = 0.0,
    out: T | None = None,
    device: DeviceType | None = None,
    return_mask: bool | None = False,
) -> T:
    """Pads a list of tensordicts in order for them to be stacked together in a contiguous format.

    Args:
        list_of_tensordicts (List[TensorDictBase]): the list of instances to pad and stack.
        batch_first (bool, optional): the ``batch_first`` correspondant of :func:`torch.nn.utils.rnn.pad_sequence`.
            Defaults to ``True``.
        padding_value (number, optional): the padding value. Defaults to ``0.0``.
        out (TensorDictBase, optional): if provided, the destination where the data will be
            written.
        device (device compatible type, optional): if provded, the device where the
            TensorDict output will be created.
        return_mask (bool, optional): if ``True``, a "mask" entry will be returned.
            It contains the mask of valid values in the stacked tensordict.

    Examples:
        >>> list_td = [
        ...     TensorDict({"a": torch.zeros((3,))}, []),
        ...     TensorDict({"a": torch.zeros((4,))}, []),
        ...     ]
        >>> padded_td = pad_sequence(list_td)
        >>> print(padded_td)
        TensorDict(
            fields={
                a: Tensor(shape=torch.Size([2, 4]), device=cpu, dtype=torch.float32, is_shared=False)},
            batch_size=torch.Size([]),
            device=None,
            is_shared=False)
    """
    if not list_of_tensordicts:
        raise RuntimeError("list_of_tensordicts cannot be empty")
    # check that all tensordict match
    if return_mask:
        list_of_tensordicts = [
            td.clone(False).set("mask", torch.ones(td.shape, dtype=torch.bool))
            for td in list_of_tensordicts
        ]
    keys = _check_keys(list_of_tensordicts, leaves_only=True, include_nested=True)
    shape = max(len(td) for td in list_of_tensordicts)
    if shape == 0:
        shape = [
            len(list_of_tensordicts),
        ]
    elif batch_first:
        shape = [len(list_of_tensordicts), shape]
    else:
        shape = [shape, len(list_of_tensordicts)]
    if out is None:
        out = TensorDict(
            {}, batch_size=torch.Size(shape), device=device, _run_checks=False
        )
        for key in keys:
            try:
                out.set(
                    key,
                    torch.nn.utils.rnn.pad_sequence(
                        [td.get(key) for td in list_of_tensordicts],
                        batch_first=batch_first,
                        padding_value=padding_value,
                    ),
                )
            except Exception as err:
                raise RuntimeError(f"pad_sequence failed for key {key}") from err
        return out
    else:
        for key in keys:
            out.set_(
                key,
                torch.nn.utils.rnn.pad_sequence(
                    [td.get(key) for td in list_of_tensordicts],
                    batch_first=batch_first,
                    padding_value=padding_value,
                ),
            )
        return out


def merge_tensordicts(*tensordicts: T) -> T:
    """Merges tensordicts together."""
    if len(tensordicts) < 2:
        raise RuntimeError(
            f"at least 2 tensordicts must be provided, got" f" {len(tensordicts)}"
        )
    d = tensordicts[0].to_dict()
    batch_size = tensordicts[0].batch_size
    for td in tensordicts[1:]:
        d.update(td.to_dict())
        if td.batch_dims < len(batch_size):
            batch_size = td.batch_size
    return TensorDict(d, batch_size, device=td.device, _run_checks=False)
