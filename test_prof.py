import torch
from torch.autograd.profiler import *
import time

evts = EventList()
for r in range(10*1000):
    st = r * 1000
    evts.append(FunctionEvent(id=0, thread=0, name="parent", start_us=st, end_us=st+100))
    evts.append(FunctionEvent(id=0, thread=0, name="parent", start_us=st+1, end_us=st+99))
    evts.append(FunctionEvent(id=0, thread=0, name="child", start_us=st+10, end_us=st+90))

st = time.time()
evts._build_tree()
print("Elapsed: {:.3f}".format(time.time() - st))

