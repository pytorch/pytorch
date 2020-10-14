call %SCRIPT_HELPERS_DIR%\setup_pytorch_env.bat

@echo on
pushd test

echo Some smoke tests
"C:\Program Files (x86)\Windows Kits\10\Debuggers\x64\gflags.exe" /i python.exe +sls
python %SCRIPT_HELPERS_DIR%\run_python_nn_smoketests.py
if ERRORLEVEL 1 exit /b 1

"C:\Program Files (x86)\Windows Kits\10\Debuggers\x64\gflags.exe" /i python.exe -sls
if ERRORLEVEL 1 exit /b 1

echo Run caffe2 ops tests
pip list
pip install pytest
cd %TMP_DIR_WIN%\build\caffe2\python\operator_test
python -m pytest -x -v --disable-warnings ^
--ignore batch_sparse_to_dense_op_test.py ^
--ignore copy_rows_to_tensor_op_test.py ^
--ignore counter_ops_test.py ^
--ignore dataset_ops_test.py ^
--ignore expand_op_test.py ^
--ignore fc_operator_test.py ^
--ignore flexible_top_k_test.py ^
--ignore index_hash_ops_test.py ^
--ignore index_ops_test.py ^
--ignore learning_rate_op_test.py ^
--ignore numpy_tile_op_test.py ^
--ignore pack_ops_test.py ^
--ignore rand_quantization_op_speed_test.py ^
--ignore roi_align_rotated_op_test.py ^
--ignore sequence_ops_test.py ^
--ignore torch_integration_test.py ^
%TMP_DIR_WIN%\build\caffe2\python\operator_test -G

if ERRORLEVEL 1 exit /b 1

popd

