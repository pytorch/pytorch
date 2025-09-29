# Example: Adding a new operator benchmark test

# To add a new test (e.g., "conv2d"), you would:

# 1. Edit operator_tests_config.yml:
# operator_benchmark_tests:
#   - matmul
#   - mm
#   - addmm
#   - bmm
#   - conv2d  # <-- Add this line

# 2. Ensure the test file exists:
# benchmarks/operator_benchmark/pt/conv2d_test.py

# 3. The test will automatically be included in the next run!

# You can verify the configuration:
# python get_config.py --tests-only
# python get_config.py --format json
