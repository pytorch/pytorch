import os


def check_env_flag(name):
    return os.getenv(name, '').upper() in ['ON', '1', 'YES', 'TRUE', 'Y']
