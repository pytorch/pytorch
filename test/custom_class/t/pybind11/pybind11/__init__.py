from ._version import version_info, __version__  # noqa: F401 imported but unused


def get_include(user=False):
    from distutils.dist import Distribution
    import os
    import sys

    # Are we running in a virtual environment?
    virtualenv = hasattr(sys, 'real_prefix') or \
        sys.prefix != getattr(sys, "base_prefix", sys.prefix)

    if virtualenv:
        return os.path.join(sys.prefix, 'include', 'site',
                            'python' + sys.version[:3])
    else:
        dist = Distribution({'name': 'pybind11'})
        dist.parse_config_files()

        dist_cobj = dist.get_command_obj('install', create=True)

        # Search for packages in user's home directory?
        if user:
            dist_cobj.user = user
            dist_cobj.prefix = ""
        dist_cobj.finalize_options()

        return os.path.dirname(dist_cobj.install_headers)
