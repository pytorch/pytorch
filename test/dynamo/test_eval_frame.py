from typing import Dict, Optional
from torch._dynamo.types import GuardedCode, GuardFn, DynamoFrameType, CacheEntry, FrameState

import inspect

def full_function_name(frame):
    # Get the function name
    function_name = frame.f_code.co_name
    
    # Check if we are inside a class
    try:
        class_name = frame.f_locals['self'].__class__.__name__
        function_name = class_name + '.' + function_name
    except KeyError:
        pass

    # Get the module name
    module = inspect.getmodule(frame)
    module_name = module.__name__ if module is not None else "empty module"

    return module_name + '.' + function_name

class Always(GuardFn):
    def __init__(self, hit=True):
        self.hit = hit

    def __call__(self, f_locals: Dict) -> bool:
        return self.hit

class MyCallback:
    def __init__(self):
        self.count = 0
    def __call__(self, frame: DynamoFrameType, cache: Optional[CacheEntry], frame_state: FrameState):
        self.count += 1
        if "count" in frame_state:
            count = frame_state["count"]
            print(f"count of last call: {count}")
        frame_state["count"] = self.count
        print(f"calling the {self.count}-th function: {full_function_name(frame)}")
        return GuardedCode(code=frame.f_code, check_fn=Always(hit=False))

callback = MyCallback()

def add(a, b):
    return a + b

import torch
from torch._dynamo.eval_frame import set_eval_frame
prior = set_eval_frame(callback)

a = add(torch.randn(50, 50), torch.randn(50, 50))
b = add(torch.randn(50, 50), torch.randn(50, 50))
print((a + b).sum().item())

set_eval_frame(prior)
