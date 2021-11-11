import torch
from torch.utils.benchmark import Language, Timer
import pickle

time_stats = Timer(
    "c = torch.add(a, b)",
    """
        a = torch.ones(1)
        b = torch.ones(1)
    """,
).collect_callgrind()#.as_standardized().stats(inclusive=False)

print (time_stats)

with open('add_before.pickle', 'wb') as before_f:
    pickle.dump(time_stats, before_f)
