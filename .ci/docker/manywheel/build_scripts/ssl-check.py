# cf. https://github.com/pypa/manylinux/issues/53

import sys
from urllib.request import urlopen


GOOD_SSL = "https://google.com"
BAD_SSL = "https://self-signed.badssl.com"


print("Testing SSL certificate checking for Python:", sys.version)

EXC = OSError

print(f"Connecting to {GOOD_SSL} should work")
urlopen(GOOD_SSL)
print("...it did, yay.")

print(f"Connecting to {BAD_SSL} should fail")
try:
    urlopen(BAD_SSL)
    # If we get here then we failed:
    print("...it DIDN'T!!!!!11!!1one!")
    sys.exit(1)
except EXC:
    print("...it did, yay.")
