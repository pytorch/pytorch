from torch.autograd.profiler import profile
from collections import namedtuple, deque
import re
import torch.nn as nn
from torchvision import models
from torch.profiler import record_function, ProfilerActivity
import torch


class Pattern:

    def match(self, event):
        raise NotImplementedError


class NamePattern(Pattern):

    def __init__(self, name):
        self.name = name

    def match(self, event):
        return re.search(self.name, event.name()) is not None


def and_(*args):

    class CompositePattern(Pattern):

        def match(self, event):
            return all(pattern.match(event) for pattern in args)

    return CompositePattern()


def or_(*args):

    class CompositePattern(Pattern):

        def match(self, event):
            return any(pattern.match(event) for pattern in args)

    return CompositePattern()


def EventTreeDFS(event_tree):
    stack = deque(event_tree)
    while stack:
        curr_event = stack.pop()
        yield curr_event
        for child_event in curr_event.children:
            stack.append(child_event)


# TODO: Think about How can we reuse the same pattern for multiple events?
def find_anti_pattern(profile, anti_pattern):
    for event in EventTreeDFS(
            profile.kineto_results.experimental_event_tree()):
        for pattern, description in anti_pattern:
            if pattern.match(event):
                print(f"{event.name()} {description}")


'''
Below Here is some preliminary test code
'''

class TestNet(nn.Module):

    def __init__(self):
        super(TestNet, self).__init__()
        self.compute1 = models.convnext_large()

    def forward(self, x):
        a = self.compute1(x)
        self.garbage_code(x)
        b = self.compute1(x)
        return a + b

    def garbage_code(self, x):
        for i in range(224):
            with record_function(f"iter#{i}"):
                x[0, 0, i, 0] = i


if __name__ == '__main__':
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    model = TestNet().to("cuda")
    inputs = torch.randn(10, 3, 224, 224).to("cuda")
    model(inputs)
    inputs = torch.randn(10, 3, 224, 224).to("cuda")
    with torch.profiler.profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            with_flops=True,
            with_modules=True) as prof:
        model(inputs)

    anti_pattern = [(or_(NamePattern('aten::add'),
                         NamePattern('aten::copy')), 'All is matched')]
    find_anti_pattern(prof.profiler, anti_pattern)