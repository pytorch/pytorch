import os

# This file copied from tools/setup_helpers/env.py
# PLEASE DO NOT ADD ANYTHING TO THIS FILE, the NAMEDTENSOR_ENABLED flag is temporary.
def check_env_flag(name, default=''):
    return os.getenv(name, default).upper() in ['ON', '1', 'YES', 'TRUE', 'Y']


def check_negative_env_flag(name, default=''):
    return os.getenv(name, default).upper() in ['OFF', '0', 'NO', 'FALSE', 'N']

NAMEDTENSOR_ENABLED = (check_env_flag('USE_NAMEDTENSOR') or
                       check_negative_env_flag('NO_NAMEDTENSOR'))
