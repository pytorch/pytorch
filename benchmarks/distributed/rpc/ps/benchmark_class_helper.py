from data.DummyData import DummyData
from models.DummyModel import DummyModel
from parameter_servers.AverageBatchParameterServer import \
    AverageBatchParameterServer
from parameter_servers.AverageParameterServer import AverageParameterServer
from trainers.DdpNcclTrainer import DdpNcclTrainer
from trainers.DdpSparseRpcTrainer import DdpSparseRpcTrainer

trainer_map = {
    "DdpNcclTrainer": DdpNcclTrainer,
    "DdpSparseRpcTrainer": DdpSparseRpcTrainer
}

ps_map = {
    "AverageParameterServer": AverageParameterServer,
    "AverageBatchParameterServer": AverageBatchParameterServer
}

model_map = {
    "DummyModel": DummyModel
}

data_map = {
    "DummyData": DummyData
}


def get_benchmark_trainer_map():
    return trainer_map


def get_benchmark_ps_map():
    return ps_map


def get_benchmark_model_map():
    return model_map


def get_benchmark_data_map():
    return data_map
