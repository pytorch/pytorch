# cf. https://github.com/pypa/manylinux/issues/53

import sys
from urllib.request import urlopen


GOOD_SSL = "https://google.com"
BAD_SSL = "https://self-signed.badssl.com"


print("Testing SSL certificate checking for Python:", sys.version)

if sys.version_info[:2] < (2, 7) or sys.version_info[:2] < (3, 4):
    print("This version never checks SSL certs; skipping tests")
    sys.exit(0)


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
