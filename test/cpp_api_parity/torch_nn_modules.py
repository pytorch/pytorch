from collections import namedtuple

# We only need to provide `cpp_sources` if we're testing an out-of-tree functional
# (e.g. `sample_functional`) and this is only done to test the C++ API parity test harness.
module_cpp_sources = {}
