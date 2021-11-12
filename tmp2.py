import torch 

# with torch.profiler.profile() as prof: 
#     a = torch.ones(2)
#     b = torch.ones(2) 
#     c = a + b 
# print(str(prof.events()))

from torch.utils.benchmark import Timer
import pickle
time_after = Timer(
    "c = torch.add(a, b)",
    """
        a = torch.ones(1)
        b = torch.ones(1)
    """,
).collect_callgrind().as_standardized().stats(inclusive=False)
print(time_after)
with open('add_after.pkl', 'wb') as after_f:
    pickle.dump(time_after, after_f)