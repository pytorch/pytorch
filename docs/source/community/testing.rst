Building PyTorch locally
------------------------
TBD: setting up your environment
Before you can test, you must build PyTorch locally::

    git checkout -b my-issue-fix
    git submodule update --init -f --recursive
    git clean -xffd; git submodule foreach --recursive git clean -xffd
    python setup.py develop
    python setup.py install

Building documentation
----------------------

Follow these steps::

    cd docs
    conda install -c conda-forge yarn nodejs matplotlib
    yarn global add katex --prefix \$CONDA_PREFIX
    pip install -r requirements.txt
    make html
    conda install 

Testing PyTorch
---------------

Since PyTorch have many layers, each layer has its own testing methods.


Python tests
~~~~~~~~~~~~

There are many tests in the ``test`` directory. Running them all is very time
consuming. You can run a single file using the `-i` flag to ``run_tests.py``::

    PYTHONPATH=. python test/run_tests.py -i test_interesting


Onnx tests
~~~~~~~~~~

Models can be saved via the onnx_ serialization protocol. This is extensively
tested via::

    export PYTHONPATH=.
    scripts/onnx/test.sh
    unset PYTHONPATH

Sometimes a PR may change the format of the serialized output, failing these
tests. You can regenerate the correct output via::

    export PYTHONPATH=.
    export EXPECTTEST_ACCEPT=1
    scripts/onnx/test.sh
    unset PYTHONPATH
    unset EXPECTTEST_ACCEPT

.. _onnx: https://onnx.ai/

CPP tests
~~~~~~~~~

C++ tests are automatically built as part of the `setup.py develop` build. There are several binaries under `./build/bin/test_*` and `./build/bin/*_test`, but since most PyTorch functionality is covered in Python tests, contributors may only need to interact with the C++ API tests in `./build/bin/test_api`. The source for C++ API tests is in `test/cpp/api/`.

The C++ tests use the Google Test framework, so the binaries provide a standardized command line interface. Run `./build/bin/test_api --help` for more information. 
