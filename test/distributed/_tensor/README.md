## Run distributed tensor tests:

from root, run (either CPU or GPU)

`pytest test/spmd/tensor/test_tensor.py`

`pytest test/spmd/tensor/test_ddp.py`

run specific test case and print stdout/stderr:

`pytest test/spmd/tensor/test_tensor.py -s -k test_tensor_from_local`
