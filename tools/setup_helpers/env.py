import os


def check_env_flag(name):
    return os.getenv(name) in ['ON', '1', 'YES', 'TRUE', 'Y']
