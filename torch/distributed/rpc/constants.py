from datetime import timedelta
from torch.distributed.constants import default_pg_timeout

# For any RpcAgent.
DEFAULT_RPC_TIMEOUT = timedelta(seconds=60)
DEFAULT_INIT_METHOD = "env://"


# For ProcessGroupAgent.
DEFAULT_NUM_SEND_RECV_THREADS = 4
# Same default timeout as in c10d.
DEFAULT_PROCESS_GROUP_TIMEOUT = default_pg_timeout
