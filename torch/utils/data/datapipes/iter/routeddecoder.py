from io import BufferedIOBase
from typing import Any, Callable, Iterable, Iterator, Sized, Tuple

from torch.utils.data import IterDataPipe, functional_datapipe
from torch.utils.data.datapipes.utils.decoder import (
    Decoder,
    basichandlers as decoder_basichandlers,
    imagehandler as decoder_imagehandler,
    extension_extract_fn
)


@functional_datapipe('decode')
class RoutedDecoderIterDataPipe(IterDataPipe[Tuple[str, Any]]):
    r""" :class:`RoutedDecoderIterDataPipe`.

    Iterable datapipe to decode binary streams from input DataPipe, yield pathname
    and decoded data in a tuple.

    Args:
        datapipe: Iterable datapipe that provides pathname and binary stream in tuples
        handlers: Optional user defined decoder handlers. If None, basic and image decoder
            handlers will be set as default. If multiple handles are provided, the priority
            order follows the order of handlers (the first handler has the top priority)
        key_fn: Function for decoder to extract key from pathname to dispatch handlers.
            Default is set to extract file extension from pathname

    Note:
        When `key_fn` is specified returning anything other than extension, the default
        handler will not work and users need to specify custom handler. Custom handler
        could use regex to determine the eligibility to handle data.
    """

    def __init__(self,
                 datapipe: Iterable[Tuple[str, BufferedIOBase]],
                 *handlers: Callable,
                 key_fn: Callable = extension_extract_fn) -> None:
        super().__init__()
        self.datapipe: Iterable[Tuple[str, BufferedIOBase]] = datapipe
        if not handlers:
            handlers = (decoder_basichandlers, decoder_imagehandler('torch'))
        self.decoder = Decoder(*handlers, key_fn=key_fn)

    def add_handler(self, *handler: Callable) -> None:
        self.decoder.add_handler(*handler)

    def __iter__(self) -> Iterator[Tuple[str, Any]]:
        for data in self.datapipe:
            pathname = data[0]
            result = self.decoder(data)
            yield (pathname, result[pathname])

    def __len__(self) -> int:
        if isinstance(self.datapipe, Sized):
            return len(self.datapipe)
        raise TypeError("{} instance doesn't have valid length".format(type(self).__name__))
