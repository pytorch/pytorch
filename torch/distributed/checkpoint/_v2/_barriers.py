import abc


class Barrier(abc.ABC):
    def __init__(self, world_size: int, timeout: int):
        self.world_size = world_size

    def wait(self, timeout):
        pass
