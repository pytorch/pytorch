import time
from collections import Counter
from typing import Any, Dict, List, Callable

from torch.fx import GraphModule
from torch.fx.experimental.tracing_analysis import TracingAnalysis
from torch.fx.node import Node


class FXProfiler(TracingAnalysis):
    """
    An example showing how to write a simple profiler analysis pass
    utilizing TracingAnalysis.

    See also PyTorch profiler:
        https://pytorch.org/tutorials/recipes/recipes/profiler.html

    Example usage:
        mod = torch.fx.symbolic_trace(...)
        prof = FXProfiler(mod)
        prof.run(*example_input)
        print(prof.summary())

    Example output:
       conv2d:49% linear:28% pixelshuffle:10% relu:7% sub:2% sigmoid:2% view:0% other:2%
    """

    def __init__(self, module: GraphModule):
        super(FXProfiler, self).__init__(module)
        # tracks how many seconds are spent in each function name
        self.times: Dict[str, float] = Counter()
        self.total = 0.0

    def run_call_any(self, node: Node, fn: Callable, args: List[Any], kwargs: Dict[str, Any]) -> Any:
        # Called to run a "call_function", "call_method", or "call_module" node
        start = time.perf_counter()
        result = super(FXProfiler, self).run_call_any(node, fn, args, kwargs)
        # consider calling `torch.cuda.synchronize()` here
        self.times[nameof(fn)] += time.perf_counter() - start
        return result

    def run(self, *args: List[Any]) -> Any:
        start = time.perf_counter()
        result = super(FXProfiler, self).run(*args)
        self.total += time.perf_counter() - start
        return result

    def summary(self, n=8):
        # generates a one-line report
        outputs = []
        other = 1.0
        for k, v in self.times.most_common(n - 1):
            v = v / self.total
            outputs.append(f"{k}:{v:.0%}")
            other -= v
        outputs.append(f"other:{other:.0%}")
        return " ".join(outputs)


class BigramCounter(TracingAnalysis):
    """
    An example showing an analysis pass in FX that looks at sliding
    windows of an op and its direct inputs and counts frequencies.

    Example usage:
        mod = torch.fx.symbolic_trace(...)
        prof = BigramCounter(mod)
        prof.run(*example_input)
        print(prof.summary(4))

    Example output:
        batchnorm2d(conv2d):53 conv2d(relu):50 relu(batchnorm2d):33 relu(add):16
    """

    def __init__(self, module: GraphModule):
        super(BigramCounter, self).__init__(module)
        # Keep track of what function names the inputs to this node came from
        self.current_node_sources: List[str] = []
        # Shadow structure to self.env the stores the name of the function that wrote it
        self.env_sources: Dict[str, str] = dict()
        # How many times we have seen each bigram
        self.counts: Dict[str, int] = Counter()

    def before_node(self, node: Node):
        # reset state at start of each node
        self.current_node_sources.clear()
        return super(BigramCounter, self).before_node(node)

    def load(self, node: Node):
        try:
            # record the function that wrote this input value
            self.current_node_sources.append(self.env_sources[node.name])
        except KeyError:
            pass  # from a placeholder or constant
        # do original behavior
        return super(BigramCounter, self).load(node)

    def run_call_any(self, node: Node, fn: Callable, args: List[Any], kwargs: Dict[str, Any]) -> Any:
        # count the bigram
        self.counts[f"{nameof(fn)}({','.join(self.current_node_sources)})"] += 1
        # tracking used to compute self.current_node_sources for the next node
        self.env_sources[node.name] = nameof(fn)
        return super(BigramCounter, self).run_call_any(node, fn, args, kwargs)

    def summary(self, n=8):
        # generates a one-line report
        return " ".join(f"{k}:{v}" for k, v in self.counts.most_common(n))


def nameof(fn: Callable):
    """ Convert a function, method, or nn.Module to a string """
    return (getattr(fn, "__name__", None) or fn.__class__.__name__).lower()
