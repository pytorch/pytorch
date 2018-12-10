from .env import check_env_flag

USE_FBGEMM = True

if check_env_flag('NO_FBGEMM'):
    USE_FBGEMM = False
else:
    # Enable FBGEMM if explicitly enabled
    if check_env_flag('USE_FBGEMM'):
        USE_FBGEMM = True
    else:
        USE_FBGEMM = False
