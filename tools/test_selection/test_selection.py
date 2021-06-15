# TODO provide function to select test-modules or test-cases within a test module / test case
# It basically loads files downloaded from tools/stats_downloader.py and convert into 
#    1. list of test_modules in test/run_test.py: TESTS 
#    2. list of test cases used for torch/testing/_internal/common_utils.py: discover_test_cases_recursively
#    3. list of test case name pattern used by unittest.main: -k argument

from typing import Dict, List

# Return list of test modules to be run
def determine_test_module(**kwargs) -> Dict[str, List[str]]:
    pass

# Return list of test cases to be run
def determine_test_case(**kwargs) -> List[str]:
    pass