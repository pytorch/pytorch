# Serialized operator test framework

Major functionality lives in `serialized_test_util.py`

## How to use
1. Extend the test case class from `SerializedTestCase`
2. Change the `@given` decorator to `@given_and_seeded`. This runs a seeded hypothesis test instance which will generate outputs if desired in addition to the unseeded hypothesis tests normally run.
3. Change a call to `unittest.main()` in `__main__` to `testWithArgs`.
4.  Run your test `python caffe2/python/operator_test/my_test.py -g` to generate serialized outputs. They will live in `caffe2/python/serialized_test/data/operator_test`, one folder per test function
5. Thereafter, runs of the test without the flag will load serialized outputs and gradient operators for comparison against the seeded run. If for any reason the seeded run's inputs are different (this can happen with different hypothesis versions or different setups), then we'll run the serialized inputs through the serialized operator to get a runtime output for comparison. 

If we'd like to extend the test framework beyond that for operator tests, we can create a new subfolder for them inside `caffe2/python/serialized_test/data`.
