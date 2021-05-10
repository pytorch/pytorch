from data.DummyData import DummyData
from models.DummyModel import DummyModel
from servers.AverageParameterServer import AverageParameterServer
from trainers.DdpNcclTrainer import DdpNcclTrainer
from trainers.DdpRpcTrainer import DdpRpcTrainer

trainer_map = {
    "DdpNcclTrainer": DdpNcclTrainer,
    "DdpRpcTrainer": DdpRpcTrainer
}

ps_map = {
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


def get_benchmark_ps_map():
    return ps_map


def get_benchmark_model_map():
    return model_map


def get_benchmark_data_map():
    return data_map
