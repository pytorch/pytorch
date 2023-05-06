from abc import ABC, abstractmethod


class Scheduling(ABC):
    @abstractmethod
    def can_fuse_vertical(self, node1, node2):
        pass

    @abstractmethod
    def can_fuse_horizontal(self, node1, node2):
        pass

    @abstractmethod
    def group_fn(self, *args, **kwargs):
        pass

    @abstractmethod
    def codegen_template(self, *args, **kwargs):
        pass

    @abstractmethod
    def codegen_nodes(self, *args, **kwargs):
        pass

    @abstractmethod
    def codegen_sync(self, *args, **kwargs):
        pass

    @abstractmethod
    def flush(self, *args, **kwargs):
        pass
