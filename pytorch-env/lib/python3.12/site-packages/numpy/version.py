
"""
Module to expose more detailed version info for the installed `numpy`
"""
version = "2.1.2"
__version__ = version
full_version = version

git_revision = "f5afe3d2ede8c1ed64cb1998cb869a4cd7831120"
release = 'dev' not in version and '+' not in version
short_version = version.split("+")[0]
