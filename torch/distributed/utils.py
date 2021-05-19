import torch

def _parse_remote_device(remote_device: str):
    r"""
    Parses the remote device.

    Args:
        remote_device (str): Device on the destination worker where we'd like to place this module.
            The format should be one of the following:

                1. "<workername>/<device>", where the device field can be parsed as torch.device type.
                   E.g., "trainer0/cpu", "trainer0", "ps0/cuda:0".
                   In addition, the device field can be optional and the default value is "cpu".
                2. "rank:<rank>/<device>", where <rank> is the rank of the
                   process and device can be parsed as torch.device type.
                   E.g., "rank:0/cpu", "rank:0", "rank:0/cuda:0"

    Returns:
        A workername/rank and a device.
    """

    PARSE_ERROR = (
        f"Could not parse remote_device: {remote_device}. The valid format is "
        "'<workername>/<device>' or 'rank:<rank>/<device>'"
    )

    fields = remote_device.split("/")
    if len(fields) == 2:
        [on, device] = fields
    elif len(fields) == 1:
        on = fields[0]
        device = "cpu"
    else:
        raise ValueError(PARSE_ERROR)

    # Since the workername in the input remote device won't be validated until the created remote module is executed,
    # only do some very basic sanity check on workername at the module creation time.
    # As currently there is no regex to describe the format of workername, just check whether the workername is empty.
    if not on:
        raise ValueError(PARSE_ERROR)

    # Validate the device.
    torch.device(device)

    # Check for rank based format
    fields = on.split(':')
    if len(fields) == 2:
        # rank:<rank>/device format, extract rank
        if fields[0] == 'rank' and fields[1].isdigit():
            on = int(fields[1])  # type: ignore[assignment]
        else:
            raise ValueError(PARSE_ERROR)
    elif len(fields) > 2:
        raise ValueError(PARSE_ERROR)

    return on, device
