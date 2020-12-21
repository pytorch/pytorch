# Example: python -m usage test/test_ops.py --verbose
# sqlite3 operators_stats.sqlite
# sqlite> select * from usage;

import pytest
import usage_checker

if __name__ == "__main__":
    pytest.main()