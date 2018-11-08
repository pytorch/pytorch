from .env import check_env_flag

if check_env_flag('NO_FBGEMM'):
    USE_FBGEMM = False
else:
    USE_FBGEMM = True
