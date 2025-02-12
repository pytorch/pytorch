import abc
from typing import Any


class CheckpointLayoutBase(abc.ABC):
    """
    This class is responsible for deciding the layout of the checkpoint on storage.

    TODO: The caller needs to ensure checkpoint loader is setup with options that are
    in sync with the options used to save the checkpoint. we can write metadata about
    the options we used to save and validate if needed? In comparison to DCP, This is
    similar to writing checkpoint with one storage writer and loading with another that
    is incompatible. so may be just document this as a requirement.
    """

    def get_file_mappings_for_write(
        self, rank: int, state_dict: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Examples usecases:

            1. To save state_dict from one rank in one file and all ranks in a single directory.
                `return {f"checkpoint_{rank}.pt": state_dict}`
            2. To save state_dict from one rank in one file and but use a separate directory
                for every 1000 ranks.
                `return {f"{rank/1000}/checkpoint_{rank}.pt": state_dict}`
            2. To save each module in a separate file.
                `
                return {
                    "model.pt": state_dict["model"],
                    "optimizer.pt": state_dict["optimizer"],
                    "dataloader.pt": state_dict["dataloader"],
                }
                `
            3. To save each param in a separate file.
                `
                return {
                    "model.weights.param1.pt": state_dict["model"]["weights"]["param1"],
                    "model.weights.param2.pt": state_dict["model"]["weights"]["param2"],
                }
                `
        """
        return {f"checkpoint_{rank}.pt": state_dict}

    @abc.abstractmethod
    def get_file_mappings_to_read(
        self, rank_info: int, fqns_to_load: list[str]
    ) -> dict[str, list[str]]:
        """ """
        pass


class FilePerRankCheckpointLayout(abc.ABC):
    """
    save checkpoint from one rank in one file and all ranks in a single directory.
    """

    def get_file_mappings_for_write(
        self, rank: int, state_dict: dict[str, Any]
    ) -> dict[str, Any]:
        return {f"checkpoint_{rank}.pt": state_dict}

    @abc.abstractmethod
    def get_file_mappings_to_read(
        self, rank: int, fqns_to_load: list[str]
    ) -> dict[str, list[str]]:
        return {f"checkpoint_{rank}.pt": fqns_to_load}
