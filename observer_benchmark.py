import torch
import time
import numpy as np
from torch.quantization.observer import HistogramObserver


#torch.manual_seed(0)
X = torch.randn(1,1,224,224)

obs = HistogramObserver(2048)
#obs = torch.quantization.MinMaxObserver()\
acc_time = 0
for i in range(100):
   X = torch.randn(10,1,320,320)
   start = time.time()
   obs(X)
   #obs.forward_new(X)
   acc_time = acc_time + time.time()-start
print(acc_time)

#(pytorch3) [raghuraman@devvm845.vll1 ~/pytorch] python observer_benchmark.py
#0.7505836486816406
#(pytorch3) [raghuraman@devvm845.vll1 ~/pytorch] python observer_benchmark.py
#12.18569564819336
