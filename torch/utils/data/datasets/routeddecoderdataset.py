from torch.utils.data.dataset import IterableDataset
from torch.utils.data.datasets.decoder import (
    Decoder,
    basichandlers as decoder_basichandlers,
    imagehandler as decoder_imagehandler)

from typing import Iterable, Iterator, Union, List

class RoutedDecoderIterableDataset(IterableDataset):
    r""" :class:`RoutedDecoderIterableDataset`.

    IterableDataset to decode binary streams from input iterables,
    yield pathname and decoded binary stream in a tuple.
    args:
        dataset: Iterable dataset that provides pathname and binary stream in tuples
        decoders: user defined decoders, if None, basic and image decoders will be set as default
        length: a nominal length of the dataset
    """

    def __init__(
            self,
            dataset : Iterable,
            *,
            decoders : Union[None, List[str]] = None,
            length: int = -1):
        super().__init__()
        self.dataset : Iterable = dataset
        if decoders:
            self.decoder = Decoder(decoders)
        else:
            self.decoder = Decoder([decoder_basichandlers, decoder_imagehandler('torch')])
        self.length : int = length

    def add_decoder(self, decoder) -> None:
        self.decoder.add_decoder(decoder)

    def __iter__(self) -> Iterator[tuple]:
        for data in self.dataset:
            pathname = data[0]
            result = self.decoder(data)
            yield (pathname, result[pathname])

    def __len__(self):
        if self.length == -1:
            raise NotImplementedError
        return self.length
