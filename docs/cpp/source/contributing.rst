Contributing to PyTorch
=======================

If you would like to contribute to the PyTorch C++ API, refer to the
`CONTRIBUTING.md
<https://github.com/pytorch/pytorch/blob/master/CONTRIBUTING.md>`_ document in
the PyTorch repository. It contains instructions on how to develop PyTorch from source
and submit a proposal for your patch or feature.

Specifically for the C++ frontend, just a note about tests: We have very
extensive tests in the `test/cpp/api
<https://github.com/pytorch/pytorch/blob/master/test/cpp/api/>`_ folder. The
tests are a great way to see how certain components are intended to be used.
Wh∑en compiling PyTorch from source, the test runner binary will be written to
``build/bin/test_api``. The tests use the `GoogleTest
<https://github.com/google/googletest/blob/master/googletest/>`_ framework,
which you can read up about to learn how to configure the test runner. When
submitting a new feature, we care very much that you write appropriate tests.
Please follow the lead of the other tests to see how to write a new test case.
