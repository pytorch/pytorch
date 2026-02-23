def xfail(op_name, variant_name="", *, device_type=None, dtypes=None):
    return (op_name, variant_name, device_type, dtypes, True)


def skip(op_name, variant_name="", *, device_type=None, dtypes=None):
    return (op_name, variant_name, device_type, dtypes, False)
