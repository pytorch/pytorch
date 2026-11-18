## Run distributed tensor tests:

from root, run (either CPU or GPU)

`pytest test/distributed/tensor/test_dtensor.py`


run specific test cases and print stdout/stderr:

`pytest test/distributed/tensor/test_dtensor.py -s -k test_from_local`
