from typing import Optional, Union

import torch


class _RemoteDevice(object):
    """
    Represents a device on a remote worker.

    Args:
        remote_device (str or int): Represents a device on a remote worker.
            The format should be one of the following:

                1. "<workername>/<device>", where the device field can be parsed as torch.device type.
                   E.g., "trainer0/cpu", "trainer0", "ps0/cuda:0".
                   In addition, the device field can be optional and the default value is "cpu".
                2. "rank:<rank>/<device>", where <rank> is the rank of the
                   process and device can be parsed as torch.device type.
                   E.g., "rank:0/cpu", "rank:0", "rank:0/cuda:0"
                3. <workername> and <rank> are optional and formats like "cpu"
                    and "cuda:1", just represent local devices.
    """

    def __init__(self, remote_device: Union[str, int]):
        PARSE_ERROR = (
            f"Could not parse remote_device: {remote_device}. The valid format is "
            "'<workername>/<device>' or 'rank:<rank>/<device>' or '<device>'"
        )
        self._remote_worker = None
        self._device: Optional[Union[str, int, torch.device]] = None

        if isinstance(remote_device, torch.device):
            self._device = remote_device
        elif isinstance(remote_device, int):
            self._device = remote_device
        else:
            fields = remote_device.split("/")
            if len(fields) == 2:
                self._remote_worker, self._device = fields
            elif len(fields) == 1:
                # Check if this is a valid device.
                if _RemoteDevice._is_valid_local_device(fields[0]):
                    self._device = fields[0]
                else:
                    self._remote_worker = fields[0]
                    self._device = "cpu"
            else:
                raise ValueError(PARSE_ERROR)

        # Do some basic sanity check (no empty string)
        if self._remote_worker is not None and not self._remote_worker:
            raise ValueError(PARSE_ERROR)

        # Validate the device.
        self._device = torch.device(self._device)

        # Check for rank based format.
        if self._remote_worker is not None:
            fields = self._remote_worker.split(":")
            if len(fields) == 2:
                # rank:<rank>/device format, extract rank
                if fields[0] == "rank" and fields[1].isdigit():
                    self._remote_worker = int(fields[1])  # type: ignore[assignment]
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

    def remote_worker(self) -> Optional[Union[int, str]]:
        """
        Returns the remote worker representing the remote device
        (could be name or rank).
        """
        return self._remote_worker

    def device(self) -> torch.device:
        """
        Returns the local device on the remote worker.
        """
        return self._device  # type: ignore[return-value]

    def __repr__(self):
        if self._remote_worker is None:
            return str(self._device)
        elif self._device is None:
            return self._remote_worker
        elif isinstance(self._remote_worker, str):
            return f"{self._remote_worker}/{self._device}"
        else:
            return f"rank:{self._remote_worker}/{self._device}"

    def __eq__(self, other):
        if not isinstance(other, _RemoteDevice):
            return False

        if (
            self._remote_worker == other._remote_worker
            and self._device == other._device
        ):
            return True

        return False
