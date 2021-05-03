class BenchmarkConfigurations:
    def __init__(
        self,
        trainer_count: int = 1,
        parameter_server_count: int = 0,
        batch_size: int = 10,
        print_metrics_to_dir: bool = False,
        master_addr: str = "localhost",
        master_port: str = "29500",
        rpc_async_timeout: int = 30,
        rpc_init_method: str = "tcp://localhost:29501"
    ):
        self.world_size = trainer_count + parameter_server_count + 1
        self.trainer_count = trainer_count
        self.parameter_server_count = parameter_server_count
        self.batch_size = batch_size
        self.print_metrics_to_dir = print_metrics_to_dir
        self.master_addr = master_addr
        self.master_port = master_port
        self.rpc_async_timeout = rpc_async_timeout
        self.rpc_init_method = rpc_init_method

    def to_string(self):
        output = ""
        class_items = list(self.__dict__.items())
        for i in range(len(class_items)):
            attr, value = class_items[i]
            if i > 0:
                output += "__"
            output += "{}_{}".format(attr, value)
        return output
