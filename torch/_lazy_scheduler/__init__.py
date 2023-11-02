import torch
import itertools
from typing import Optional, Dict, Callable


class Segment:
  _cur_segment: Optional[str] = None
  _func_to_segment_mapping: Dict[Callable, str] = {}
  _unnamed_counter = itertools.count(start=0)

  @classmethod
  def get_next_unnamed_segment(cls):
    return f"unnamed_{next(cls._unnamed_counter)}"


class LazyScheduler:
  def __init__(self, schedule):
    self.schedule = schedule

  def compile(self, gm_, example_inputs_):
    # breakpoint()
    known_segments = []
    print(f"gm_.graph: {gm_.graph}")
    for node in gm_.graph.nodes:
      print(f"node: {node}")
      print(f"node.meta['segment']: {node.meta['segment']}")
      if len(known_segments) == 0 or node.meta["segment"] != known_segments[-1]:
        known_segments.append(node.meta["segment"])

    def split_callback(node):
      return known_segments.index(node.meta["segment"])

    qualname_map = {}
    gm_after_split = torch.fx.passes.split_module.split_module(
      m=gm_,
      root_m=None,
      split_callback=split_callback,
      qualname_map=qualname_map,
      keep_original_order=True,
    )
    gm__node_list = list(gm_.graph.nodes)
    gm_after_split_node_list = list(gm_after_split.graph.nodes)
    # submod_list = list(gm_after_split.children())
    # breakpoint()

    # TODO: How do we generate example inputs for each compiled subgraph?

    # assert isinstance(root, FakeRootModule)
    # gm = fx.GraphModule(root, self.graph)

    return torch._inductor.compile(gm_, example_inputs_)

"""
FAQ

Q1: What happens if we have a user-defined segment deep down in a submodule?
Answer: everything before the defined segment will be in their own segment. Everything after is in another segment.
You can call this a "segment break".
"""
