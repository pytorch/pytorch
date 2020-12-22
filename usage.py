# Example: python -m usage test/test_ops.py --verbose
# sqlite3 operators_stats.sqlite
# sqlite> select * from usage;

# ls test/test_*.py | perl -ne 'chomp; print "python -m usage $_ --verbose\n"' | bash

import pytest
import usage_checker

if __name__ == "__main__":
    pytest.main()
    usage_checker.flush()
