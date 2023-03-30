from typing import Optional, Union

import torch


class _remote_device:
    """
    Represents a device on a remote worker.

    Args:
        remote_device (str or torch.device): Represents a device on a remote worker.
            The string format should be one of the following:

                1. "<workername>/<device>", where the device field can be parsed as torch.device type.
                   E.g., "trainer0/cpu", "trainer0", "ps0/cuda:0".
                   In addition, the device field can be optional and the default value is "cpu".
                2. "rank:<rank>/<device>", where <rank> is the rank of the
                   process and device can be parsed as torch.device type.
                   E.g., "rank:0/cpu", "rank:0", "rank:0/cuda:0"
                3. <workername> and <rank> are optional and formats like "cpu"
                    and "cuda:1", just represent local devices.
    """

    def __init__(self, remote_device: Union[str, torch.device]):
        PARSE_ERROR = (
            f"Could not parse remote_device: {remote_device}. The valid format is "
            "'<workername>/<device>' or 'rank:<rank>/<device>' or '<device>'"
        )
        self._worker_name = None
        self._rank = None
        self._device: Optional[Union[str, int, torch.device]] = None

        if isinstance(remote_device, torch.device):
            self._device = remote_device
        elif isinstance(remote_device, str):
            fields = remote_device.split("/")
            if len(fields) == 2:
                self._worker_name, self._device = fields
            elif len(fields) == 1:
                # Check if this is a valid device.
                if _remote_device._is_valid_local_device(fields[0]):
                    self._device = fields[0]
                else:
                    self._worker_name = fields[0]
                    self._device = "cpu"
            else:
                raise ValueError(PARSE_ERROR)
        else:
            raise TypeError(f'Invalid type for remote_device: {type(remote_device)}')

        # Do some basic sanity check (no empty string)
        if self._worker_name is not None and not self._worker_name:
            raise ValueError(PARSE_ERROR)

        # Validate the device.
        self._device = torch.device(self._device)

        # Check for rank based format.
        if self._worker_name is not None:
            fields = self._worker_name.split(":")
            if len(fields) == 2:
                # rank:<rank>/device format, extract rank
                if fields[0] == "rank" and fields[1].isdigit():
                    self._rank = int(fields[1])  # type: ignore[assignment]
                    self._worker_name = None
                else:
                    raise ValueError(PARSE_ERROR)
            elif len(fields) > 2:
                raise ValueError(PARSE_ERROR)

    @staticmethod
    def _is_valid_local_device(device):
        # Check for torch.device
        try:
            torch.device(device)
            return True
        except Exception:
            return False

    def worker_name(self) -> Optional[str]:
        """
        Returns the name of remote worker representing the remote device.
        Returns ``None`` if no worker name is available.
        """
        return self._worker_name

    def rank(self) -> Optional[int]:
        """
        Returns the rank of remote worker representing the remote device.
        Returns ``None`` if no rank is available.
        """
        return self._rank

    def device(self) -> torch.device:
        """
        Returns the local device on the remote worker.
        """
        return self._device  # type: ignore[return-value]

    def __repr__(self):
        if self._device is not None:
            if self._worker_name is not None:
                return f'{self._worker_name}/{self._device}'
            elif self._rank is not None:
                return f'rank:{self._rank}/{self._device}'
            else:
                return str(self._device)
        else:
            if self._worker_name is not None:
                return f'{self._worker_name}'
            elif self._rank is not None:
                return f'{self._rank}'
            else:
                raise RuntimeError('Invalid state!')

    def __eq__(self, other):
        if not isinstance(other, _remote_device):
            return False

        if (
            self._worker_name == other._worker_name
            and self._device == other._device
            and self._rank == other._rank
        ):
            return True

        return False


    def __hash__(self):
        return hash(self._worker_name) ^ \
            hash(self._device) ^ \
            hash(self._rank)
