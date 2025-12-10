
"""
Module to expose more detailed version info for the installed `numpy`
"""
version = "2.3.5"
__version__ = version
full_version = version

git_revision = "c3d60fc8393f3ca3306b8ce8b6453d43737e3d90"
release = 'dev' not in version and '+' not in version
short_version = version.split("+")[0]
