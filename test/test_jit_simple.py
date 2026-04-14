# Owner(s): ["oncall: jit"]

import sys
sys.argv.append("--jit-executor=simple")
from torch.testing._internal.common_utils import parse_cmd_line_args
parse_cmd_line_args()
from test_jit import *  # noqa: F403

if __name__ == '__main__':
    if sys.version_info < (3, 14):
        run_tests()
