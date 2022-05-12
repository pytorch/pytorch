def fb_xplat_genrule(default_outs = ["."], **kwargs):
    native.genrule(
        # default_outs=default_outs, # only needed for internal BUC
        **kwargs
    )
