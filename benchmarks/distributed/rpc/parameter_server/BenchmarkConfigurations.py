from dataclasses import dataclass


@dataclass
class BenchmarkConfigurations:
    trainer_count: int = 1
    parameter_server_count: int = 0
    batch_size: int = 1
    print_metrics_to_dir: bool = False
    master_addr: str = "localhost"
    master_port: str = "29500"
    rpc_async_timeout: int = 5
    rpc_init_method: str = "tcp://localhost:29501"
