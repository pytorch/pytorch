from torch._dynamo import allow_in_graph
from torch.distributed._tensor.api import DTensor

# dynamo/torch.compile utils for
allow_in_graph(DTensor)
allow_in_graph(DTensor.from_local)
