from data.DummyData import DummyData
from models.DummyModel import DummyModel
from servers.AverageParameterServer import AverageParameterServer
from trainers.DdpNcclTrainer import DdpNcclTrainer
from trainers.DdpSparseRpcTrainer import DdpSparseRpcTrainer
from trainers.DdpTrainer import DdpTrainer

trainer_map = {
    "DdpNcclTrainer": DdpNcclTrainer,
    "DdpTrainer": DdpTrainer,
    "DdpSparseRpcTrainer": DdpSparseRpcTrainer
}

server_map = {
    "AverageParameterServer": AverageParameterServer
}

model_map = {
    "DummyModel": DummyModel
}

data_map = {
    "DummyData": DummyData
}


def get_benchmark_trainer_map():
    return trainer_map


def get_benchmark_server_map():
    return server_map


def get_benchmark_model_map():
    return model_map


def get_benchmark_data_map():
    return data_map
