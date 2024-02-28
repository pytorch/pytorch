from .server import AverageBatchParameterServer, AverageParameterServer

server_map = {
    "AverageParameterServer": AverageParameterServer,
    "AverageBatchParameterServer": AverageBatchParameterServer,
}
