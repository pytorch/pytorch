major = 2

try:
    from __svn_version__ import version
    version_info = (major, version)
    version = '%s_%s' % version_info
except (ImportError, ValueError):
    version = str(major)
