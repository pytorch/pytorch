def _parse_remote_device(remote_device: str):
    r"""
    Parses the remote device.

    Args:
        remote_device (str): Device on the destination worker where we‘d like to place this module.
            The format should be "<workername>/<device>", where the device field can be parsed as torch.device type.
            E.g., "trainer0/cpu", "trainer0", "ps0/cuda:0".
            In addition, the device field can be optional and the default value is "cpu".

    Returns:
        A workername and a device.
    """
    fields = remote_device.split("/")
    if len(fields) == 2:
        [on, device] = fields
    elif len(fields) == 1:
        on = fields[0]
        device = "cpu"
    else:
        raise RuntimeError(
            "Could not parse remote_device: {}. The valid format is '<workername>/<device>'".format(
                remote_device
            )
        )

    # Since the workername in the input remote device won't be validated until the created remote module is executed,
    # only do some very basic sanity check on workername at the module creation time.
    # As currently there is no regex to describe the format of workername, just check whether the workername is empty.
    if not on:
        raise RuntimeError(
            "The workername in remote_device '{}' cannot be empty. The valid format is '<workername>/<device>'".format(
                remote_device
            )
        )

    return on, device
