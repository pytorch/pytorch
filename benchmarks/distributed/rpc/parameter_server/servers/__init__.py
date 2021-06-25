from .AverageBatchParameterServer import AverageBatchParameterServer
from .AverageParameterServer import AverageParameterServer

server_map = {
    "AverageParameterServer": AverageParameterServer,
    "AverageBatchParameterServer": AverageBatchParameterServer
}
