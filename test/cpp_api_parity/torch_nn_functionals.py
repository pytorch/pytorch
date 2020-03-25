from collections import namedtuple

# We only need to provide `cpp_sources` if we're testing an out-of-tree module
# (e.g. `SampleModule`) and this is only done to test the C++ API parity test harness.
functional_cpp_sources = {}
