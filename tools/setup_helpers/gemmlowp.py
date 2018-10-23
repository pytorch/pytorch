from .env import check_env_flag

if check_env_flag('NO_GEMMLOWP'):
    USE_GEMMLOWP = False
else:
    USE_GEMMLOWP = True
