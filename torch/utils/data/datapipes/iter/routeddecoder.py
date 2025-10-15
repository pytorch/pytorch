from collections.abc import Callable, Iterable, Iterator, Sized
from io import BufferedIOBase
from typing import Any

from torch.utils.data.datapipes._decorator import functional_datapipe
from torch.utils.data.datapipes.datapipe import IterDataPipe
from torch.utils.data.datapipes.utils.common import _deprecation_warning
from torch.utils.data.datapipes.utils.decoder import (
    basichandlers as decoder_basichandlers,
    Decoder,
    extension_extract_fn,
    imagehandler as decoder_imagehandler,
)


__all__ = ["RoutedDecoderIterDataPipe"]


@functional_datapipe("routed_decode")
class RoutedDecoderIterDataPipe(IterDataPipe[tuple[str, Any]]):
    r"""
    Decodes binary streams from input DataPipe, yields pathname and decoded data in a tuple.

    (functional name: ``routed_decode``)

    Args:
        datapipe: Iterable datapipe that provides pathname and binary stream in tuples
        handlers: Optional user defined decoder handlers. If ``None``, basic and image decoder
            handlers will be set as default. If multiple handles are provided, the priority
            order follows the order of handlers (the first handler has the top priority)
        key_fn: Function for decoder to extract key from pathname to dispatch handlers.
            Default is set to extract file extension from pathname

    Note:
        When ``key_fn`` is specified returning anything other than extension, the default
        handler will not work and users need to specify custom handler. Custom handler
        could use regex to determine the eligibility to handle data.
    """

    def __init__(
        self,
        datapipe: Iterable[tuple[str, BufferedIOBase]],
        *handlers: Callable,
        key_fn: Callable = extension_extract_fn,
    ) -> None:
        super().__init__()
        self.datapipe: Iterable[tuple[str, BufferedIOBase]] = datapipe
        if not handlers:
            handlers = (decoder_basichandlers, decoder_imagehandler("torch"))
        self.decoder = Decoder(*handlers, key_fn=key_fn)
        _deprecation_warning(
            type(self).__name__,
            deprecation_version="1.12",
            removal_version="1.13",
            old_functional_name="routed_decode",
        )

    def add_handler(self, *handler: Callable) -> None:
        self.decoder.add_handler(*handler)

    def __iter__(self) -> Iterator[tuple[str, Any]]:
        for data in self.datapipe:
            pathname = data[0]
            result = self.decoder(data)
            yield (pathname, result[pathname])

    def __len__(self) -> int:
        if isinstance(self.datapipe, Sized):
            return len(self.datapipe)
        raise TypeError(f"{type(self).__name__} instance doesn't have valid length")
