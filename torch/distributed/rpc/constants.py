from datetime import timedelta


# For any RpcAgent.
DEFAULT_RPC_TIMEOUT = timedelta(seconds=60)
DEFAULT_INIT_METHOD = "env://"


# For ProcessGroupAgent.
DEFAULT_NUM_SEND_RECV_THREADS = 4
