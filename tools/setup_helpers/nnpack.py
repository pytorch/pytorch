from .env import check_env_flag

if check_env_flag('NO_NNPACK'):
    WITH_NNPACK = False
else:
    WITH_NNPACK = True
