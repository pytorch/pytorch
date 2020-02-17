from datetime import timedelta
from torch.distributed.distributed_c10d import _default_pg_timeout

# For any RpcAgent.
DEFAULT_RPC_TIMEOUT = timedelta(seconds=60)
DEFAULT_INIT_METHOD = "env://"


# For ProcessGroupAgent.
DEFAULT_NUM_SEND_RECV_THREADS = 4
# Same default timeout as in c10d.
DEFAULT_PROCESS_GROUP_TIMEOUT = _default_pg_timeout
