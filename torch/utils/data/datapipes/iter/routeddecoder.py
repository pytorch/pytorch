from io import BufferedIOBase
from typing import Any, Callable, Iterable, Iterator, Sized, Tuple

from torch.utils.data import IterDataPipe, functional_datapipe
from torch.utils.data.datapipes.utils.decoder import (
    Handler, Decoder,
    basichandlers as decoder_basichandlers,
    imagehandler as decoder_imagehandler,
    _default_key_fn)


@functional_datapipe('decode')
class RoutedDecoderIterDataPipe(IterDataPipe[Tuple[str, Any]]):
    r""" :class:`RoutedDecoderIterDataPipe`.

    Iterable datapipe to decode binary streams from input DataPipe, yield pathname
    and decoded data in a tuple.
    args:
        datapipe: Iterable datapipe that provides pathname and binary stream in tuples
        handlers: Optional user defined decoder handlers. If None, basic and image decoder
            handlers will be set as default. If multiple handles are provided, the priority
            order follows the order of handlers (the first handler has the top priority)
        key_fn: Function for decoder to extract key from pathname to dispatch handlers.
            Default is set to extract file extension from pathname
    """

    def __init__(self,
                 datapipe: Iterable[Tuple[str, BufferedIOBase]],
                 *handler: Handler,
                 key_fn: Callable = _default_key_fn) -> None:
        super().__init__()
        self.datapipe: Iterable[Tuple[str, BufferedIOBase]] = datapipe
        if not handler:
            handler = (*decoder_basichandlers, decoder_imagehandler('torch'))
        self.decoder = Decoder(*handler, key_fn=key_fn)

    def add_handler(self, *handler: Handler) -> None:
        self.decoder.add_handler(*handler)

    def __iter__(self) -> Iterator[Tuple[str, Any]]:
        for data in self.datapipe:
            pathname = data[0]
            result = self.decoder(data)
            yield (pathname, result[pathname])

    def __len__(self) -> int:
        if isinstance(self.datapipe, Sized):
            return len(self.datapipe)
        raise NotImplementedError
