# pyre-strict
from collections.abc import Iterable, Iterator
from typing import Generic, TypeVar, Union

import torch


_T = TypeVar("_T")


class ProxyValue(Generic[_T]):
    # pyre-ignore
    def __init__(self, data: Iterable[_T], proxy: Union[torch.fx.Proxy, torch.fx.Node]):
        # pyre-ignore
        self.data = data
        self.proxy_or_node = proxy

    @property
    def node(self) -> torch.fx.Node:
        if isinstance(self.proxy_or_node, torch.fx.Node):
            return self.proxy_or_node
        assert isinstance(self.proxy_or_node, torch.fx.Proxy)
        return self.proxy_or_node.node

    @property
    def proxy(self) -> torch.fx.Proxy:
        if not isinstance(self.proxy_or_node, torch.fx.Proxy):
            raise RuntimeError(
                f"ProxyValue doesn't have attached Proxy object. Node: {self.proxy_or_node.format_node()}"
            )
        return self.proxy_or_node

    def to_tensor(self) -> torch.Tensor:
        assert isinstance(self.data, torch.Tensor)
        return self.data

    def is_tensor(self) -> bool:
        return isinstance(self.data, torch.Tensor)

    # pyre-ignore
    def __iter__(self) -> Iterator[_T]:
        yield from self.data

    def __bool__(self) -> bool:
        return bool(self.data)
