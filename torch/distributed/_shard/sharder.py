import abc
import torch.nn as nn

class Sharder(abc.ABC):
    @abc.abstractmethod
    def shard(self, module: nn.Module) -> nn.Module:
        """
        Processes a module and if needed swaps it with a custom sharded
        Implementation. Should return ``None`` if no swapping should be
        performed.

        The Sharder would produce ShardedTensors for the module based on
        ShardingPlan, and then call the ShardedModuleSwapper. The passed
        in module would consist of ShardedTensors and a common way to
        perform module swapping would be to use the state_dict of the passed
        in module and apply it to the new sharded module via its
        load_state_dict method.
        """
        pass
