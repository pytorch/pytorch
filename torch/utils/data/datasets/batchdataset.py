import torch
from torch.utils.data import IterableDataset
from typing import TypeVar, Optional, Iterator, List, Sized, Union, Any, Sequence, Mapping

T_co = TypeVar('T_co', covariant=True)
PaddedShapeType = Union[int, Sequence[int], Sequence[Sequence[int]], torch.Size, Sequence[torch.Size]]
PaddedValueType = Union[Any, Sequence[Any]]


class BatchIterableDataset(IterableDataset[List[T_co]]):
    r""" :class:`BatchIterableDataset`.

    IterableDataset to create mini-batches of data. An outer dimension will be added as
    `batch_size` if `drop_last` is set to `True`, or `length % batch_size` for the
    last batch if `drop_last` is set to `False`.
    args:
        dataset: IterableDataset being batched
        batch_size: The size of each batch
        drop_last: Option to drop the last batch if it's not full
    """
    dataset: IterableDataset[T_co]
    batch_size: int
    drop_last: bool
    length: Optional[int]

    def __init__(self,
                 dataset: IterableDataset[T_co],
                 *,
                 batch_size: int,
                 drop_last: bool = False,
                 ) -> None:
        assert batch_size > 0, "Batch size is required to be larger than 0!"
        super(BatchIterableDataset, self).__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.length = None

    def __iter__(self) -> Iterator[List[T_co]]:
        batch: List[T_co] = []
        for x in self.dataset:
            batch.append(x)
            if len(batch) == self.batch_size:
                yield batch
                batch.clear()
        if len(batch) > 0:
            if not self.drop_last:
                yield batch
            batch.clear()

    def __len__(self) -> int:
        if self.length is not None:
            return self.length
        if isinstance(self.dataset, Sized) and len(self.dataset) >= 0:
            if self.drop_last:
                self.length = len(self.dataset) // self.batch_size
            else:
                self.length = (len(self.dataset) + self.batch_size - 1) // self.batch_size
            return self.length
        raise NotImplementedError


class PaddedBatchIterableDataset(IterableDataset[List[T_co]]):
    r""" :class:`PaddedBatchIterableDataset`.

    IterableDataset to create padded mini-batches of data. An outer dimension will be added as
    `batch_size` if `drop_last` is set to `True`, or `length % batch_size` for the
    last batch if `drop_last` is set to `False`.

    Each item will be padded out to the corresponding `padded_shape` with `padded_value` prior
    to batch. For the dimension in `padded_shape` is `None`, item will be padded out to the
    maximum length along that dimension. If `padded_value` is `None`, padded value will be
    determined by the element type of input dataset.

    args:
        dataset: IterableDataset being padded and batched
        batch_size: The size of each batch
        padded_shapes: The nested shape(s) of each element to be padded prior to batch. None or
            unknown dimensions will be padded to the maximum size in the batch
        padded_values: The nested value(s) of each element to be padded with. None value will
            be converted to default value based on the input element.
        drop_last: Option to drop the last batch if it's not full
    """
    dataset: IterableDataset[T_co]
    batch_size: int
    padded_shapes: Optional[PaddedShapeType]
    padded_values: Optional[PaddedValueType]
    drop_last: bool
    length: Optional[int]

    def __init__(self,
                 dataset: IterableDataset[T_co],
                 *,
                 batch_size: int,
                 padded_shapes: Optional[PaddedShapeType] = None,
                 padded_values: Optional[PaddedValueType] = None,
                 drop_last: bool = False,
                 ) -> None:
        assert batch_size > 0, "Batch size is required to be larger than 0!"
        super(PaddedBatchIterableDataset, self).__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.padded_shapes = padded_shapes
        self.padded_values = padded_values
        self.drop_last = drop_last
        self.length = None

    @staticmethod
    def _get_max_size(batch):
        if isinstance(batch[0], (float, int, complex, str, Sequence, Mapping)):
            return None

        if isinstance(batch[0], torch.Tensor):
            dims = len(batch[0].shape)
            if not all(len(e.shape) == dims for e in batch):
                raise RuntimeError("Each tensor element in batch should be of equal "
                                   "dimensions.")
            shapes = [tuple(elem.shape) for elem in batch]
            # Find maximum size for each dimension
            return torch.Size([max(s) for s in zip(*shapes)])

        raise RuntimeError("The padded batch doesn't support the element type {}"
                           .format(type(batch[0])))

    # All elements share one shape/value in the batch
    # It's useful for a list of Sequence/Mapping
    @staticmethod
    def _broadcast_batch(batch, shape_value):
        if isinstance(shape_value, torch.Size) or not isinstance(shape_value, Sequence):
            while True:
                yield shape_value
        # Broadcast
        elif len(shape_value) == 1:
            while True:
                yield shape_value[0]
        else:
            if not all(len(e) == len(shape_value) for e in batch):
                raise RuntimeError("The length of all element list should be equal "
                                   "to the length of the padded shape/value {}({})."
                                   .format(shape_value, len(shape_value)))
            yield from shape_value

    @staticmethod
    def _transpose_seq(batch, shape, value):
        elem_size = len(batch[0])
        if not all(len(e) == elem_size for e in batch):
            raise RuntimeError("Each list element in batch should be of equal size.")

        # Broadcast
        shape = PaddedBatchIterableDataset._broadcast_batch(batch, shape)
        value = PaddedBatchIterableDataset._broadcast_batch(batch, value)
        return zip(*batch, shape, value)

    @staticmethod
    def _transpose_map(batch, shape, value):
        elem_keys = batch[0].keys()
        if not all(e.keys() == elem_keys for e in batch):
            raise RuntimeError("Each map element in batch should be of same keys.")

        # Broadcast
        shape = PaddedBatchIterableDataset._broadcast_batch(batch, shape)
        value = PaddedBatchIterableDataset._broadcast_batch(batch, value)
        return zip(elem_keys, shape, value)

    @staticmethod
    def _pad_tensor(batch, shape, value):
        # Change integer/sequence to torch.Size
        if not isinstance(shape, torch.Size):
            if isinstance(shape, int):
                shape = [shape]
            elif isinstance(shape, Sequence):
                if not all(isinstance(s, int) for s in shape):
                    raise RuntimeError("The padded shape {} should be a one-dimensional "
                                       "list/tuple of integers.".format(shape))
            else:
                raise RuntimeError("Invalid padded shape {}.".format(shape))
            shape = torch.Size(shape)

        if isinstance(value, Sequence) and len(value) == 1:
            value = value[0]

        new_batch = []
        for elem in batch:
            if elem.shape != torch.Size([]) and len(elem.shape) != len(shape):
                raise RuntimeError("The padded shape {} should have same number of "
                                   "dimensions as the shape of the element {}."
                                   .format(list(shape), list(elem.shape)))

            if elem.shape == shape:
                new_batch.append(elem)
            else:
                v = 0 if value is None else value
                new_elem = elem.new_full(shape, v)
                slices = None
                if elem.shape == torch.Size([]):
                    slices = [slice(0, 1) for i in range(len(shape))]
                else:
                    slices = [slice(0, n) for n in elem.shape]
                # Raises RuntimeError for the shape of element larger than
                # the padded shape on any dimension
                try:
                    new_elem[slices] = elem
                except RuntimeError:
                    raise RuntimeError("Tensor can not be padded to the target shape. "
                                       "Target: {}; Tensor: {}"
                                       .format(list(shape), list(elem.shape)))
                # As in-place operation can not be used for leaf variable,
                # convert new element to requires_grad after the operation
                if elem.requires_grad:
                    new_elem.requires_grad_()
                new_batch.append(new_elem)
        return new_batch

    # Recursive pad element into shape with value
    @staticmethod
    def _pad(batch, shape: Optional[PaddedShapeType], value: Optional[PaddedValueType]):
        elem = batch[0]
        elem_type = type(elem)
        if not all(isinstance(elem, elem_type) for elem in batch):
            raise RuntimeError("Each element in the list of batch should be same type.")

        # Find largest size along each dimension
        if shape is None:
            shape = PaddedBatchIterableDataset._get_max_size(batch)

        if isinstance(elem, torch.Tensor):
            return PaddedBatchIterableDataset._pad_tensor(batch, shape, value)
        # Do not apply padding to float/integer/complex number and string
        elif isinstance(elem, (float, int, complex, str)):
            return batch
        elif isinstance(elem, Sequence):
            transposed = PaddedBatchIterableDataset._transpose_seq(batch, shape, value)
            return [PaddedBatchIterableDataset._pad(e, s, v) for *e, s, v in transposed]
        elif isinstance(elem, Mapping):
            transposed = PaddedBatchIterableDataset._transpose_map(batch, shape, value)
            return {k: PaddedBatchIterableDataset._pad([e[k] for e in batch], s, v) for k, s, v in transposed}
        else:
            raise RuntimeError("The padded batch doesn't support the element type {}"
                               .format(type(elem)))

    def __iter__(self) -> Iterator[List[T_co]]:
        batch: List[T_co] = []
        for x in self.dataset:
            batch.append(x)
            if len(batch) == self.batch_size:
                yield PaddedBatchIterableDataset._pad(batch, self.padded_shapes, self.padded_values)
                batch.clear()
        if len(batch) > 0:
            if not self.drop_last:
                yield PaddedBatchIterableDataset._pad(batch, self.padded_shapes, self.padded_values)
            batch.clear()

    def __len__(self) -> int:
        if self.length is not None:
            return self.length
        if isinstance(self.dataset, Sized) and len(self.dataset) >= 0:
            if self.drop_last:
                self.length = len(self.dataset) // self.batch_size
            else:
                self.length = (len(self.dataset) + self.batch_size - 1) // self.batch_size
            return self.length
        raise NotImplementedError
