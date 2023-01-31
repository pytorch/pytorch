#!/bin/bash
cd "$(dirname "${BASH_SOURCE[0]}")"
./bin/nvfuser_tests
python python_tests/test_dynamo.py
python python_tests/test_python_frontend.py
PYTORCH_TEST_WITH_SLOW=1 python python_tests/test_torchscript.py
