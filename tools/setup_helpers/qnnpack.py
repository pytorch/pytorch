from .env import check_env_flag

if check_env_flag('NO_QNNPACK'):
    USE_QNNPACK = False
else:
    USE_QNNPACK = True
