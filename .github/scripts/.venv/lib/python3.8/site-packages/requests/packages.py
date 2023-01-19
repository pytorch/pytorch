import sys

try:
    import chardet
except ImportError:
    import charset_normalizer as chardet
    import warnings

    warnings.filterwarnings('ignore', 'Trying to detect', module='charset_normalizer')

# This code exists for backwards compatibility reasons.
# I don't like it either. Just look the other way. :)

for package in ('urllib3', 'idna'):
    locals()[package] = __import__(package)
    # This traversal is apparently necessary such that the identities are
    # preserved (requests.packages.urllib3.* is urllib3.*)
    for mod in list(sys.modules):
        if mod == package or mod.startswith(package + '.'):
            sys.modules['requests.packages.' + mod] = sys.modules[mod]

target = chardet.__name__
for mod in list(sys.modules):
    if mod == target or mod.startswith(target + '.'):
        sys.modules['requests.packages.' + target.replace(target, 'chardet')] = sys.modules[mod]
# Kinda cool, though, right?
