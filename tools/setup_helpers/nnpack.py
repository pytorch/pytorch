from .env import check_env_flag

if check_env_flag('NO_NNPACK'):
    USE_NNPACK = False
else:
    USE_NNPACK = True
