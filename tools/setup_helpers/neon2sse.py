from .env import check_env_flag
if check_env_flag('NO_NEON2SSE'):
    USE_NEON2SSE = False
else:
    USE_NEON2SSE = True
