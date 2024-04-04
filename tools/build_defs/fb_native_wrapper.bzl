# Only used for PyTorch open source BUCK build
# @lint-ignore-every BUCKRESTRICTEDSYNTAX
def _genrule(default_outs = ["."], **kwargs):
    if read_config("pt", "is_oss", "0") == "0":
        fail("This file is for open source pytorch build. Do not use it in fbsource!")

    genrule(
        # default_outs is only needed for internal BUCK
        **kwargs
    )

def _read_config(**kwargs):
    read_config(**kwargs)

def _filegroup(**kwargs):
    filegroup(**kwargs)

fb_native = struct(
    genrule = _genrule,
    read_config = _read_config,
    filegroup = _filegroup,
)
