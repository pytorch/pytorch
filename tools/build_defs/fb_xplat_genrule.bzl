# Only used for PyTorch open source BUCK build
# @lint-ignore-every BUCKRESTRICTEDSYNTAX

def fb_xplat_genrule(default_outs = ["."], apple_sdks = None, **kwargs):
    if read_config("pt", "is_oss", "0") == "0":
        fail("This file is for open source pytorch build. Do not use it in fbsource!")

    genrule(
        # default_outs=default_outs, # only needed for internal BUCK
        **kwargs
    )
