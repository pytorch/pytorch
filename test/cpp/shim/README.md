# Shim Tests

All shim functions expected to be used by a custom extension library should be tested end to end in
`test/cpp_extensions/test_libtorch_agnostic.py`.
Some C shims are not so directly called, for example C shims that are meant for lower level usage (e.g., binding to rust).
This directory is intended for tests that require finer grained testing than what can be provided with the end to end tests.

Whenever making changes, verify valgrind runs clean against the tests with:
```
./test/cpp/shim/run_test.sh ./build/bin/
```

Tests relying only on the headeronly parts should go into `aoti_abi_check` instead.
