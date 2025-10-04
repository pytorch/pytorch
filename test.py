import torch
from torch import nn
from torch.fx.graph_module import _forward_from_src
import torch.profiler
from collections import defaultdict

class MLPModule(nn.Module):
    def __init__(self, device, bias=True):
        super().__init__()
        torch.manual_seed(5)
        self.net1 = nn.Linear(1000, 1600, bias=bias, device=device)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(1600, 1000, bias=bias, device=device)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))

inp = torch.rand(1000).to("cuda")
ep = torch.export.export(MLPModule("cuda"), (torch.rand(1000).to("cuda"),))

gm = ep.module()

# 1) direct call
gm(inp)



import json


# # 2) codegen and then call
python_code = gm.graph.python_code(root_module="self", verbose=True, record_func=True)
co_fields = gm._graph._co_fields if hasattr(gm._graph, "_co_fields") else {}
print(python_code.src)
fn = _forward_from_src(python_code.src, python_code.globals, co_fields)
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    # schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
) as prof:
    fn(gm, inp)
prof.export_chrome_trace("trace.json")
traced_data = json.load(open("trace.json", 'r'))
trace_events = traced_data.get("traceEvents", [])

def map_recorded_events_to_aten_ops_with_stack_trace(graph_module, trace_events):
    """
    Maps recorded profiler events to their corresponding aten operations and adds stack traces.

    Args:
        graph_module: The FX GraphModule
        trace_events: List of profiler events from Chrome trace

    Returns:
        Dict mapping recorded event names to their aten operations with added stack traces
    """
    # Create a mapping from node name to node for easy lookup
    node_map = {node.name: node for node in graph_module.graph.nodes}


    # Find aten operation events
    aten_events = [e for e in trace_events if e.get("cat") == "cpu_op"]

    # Map recorded events to aten ops and add stack traces
    event_mapping = {}

    for recorded_event in trace_events:
        if (recorded_event.get("cat") == "user_annotation" and
            recorded_event.get("name", "").startswith("## ") and
            recorded_event.get("name", "").endswith(" ##")):
            # Extract node name from "## node_name ##"
            node_name = recorded_event["name"][3:-3]  # Remove "## " and " ##"

            if node_name in node_map:
                node = node_map[node_name]

                # Find corresponding aten operations within this recorded event's time window
                recorded_start = recorded_event["ts"]
                recorded_end = recorded_start + recorded_event["dur"]

                # Find aten ops that fall within this time window
                corresponding_aten_ops = []
                for aten_event in aten_events:
                    aten_start = aten_event["ts"]
                    aten_end = aten_start + aten_event["dur"]

                    # Check if aten event overlaps with recorded event
                    if (aten_start >= recorded_start and aten_start <= recorded_end) or \
                    (aten_end >= recorded_start and aten_end <= recorded_end) or \
                    (aten_start <= recorded_start and aten_end >= recorded_end):
                        corresponding_aten_ops.append(aten_event)

                # Add stack trace to recorded event and aten ops
                stack_trace = node.meta.get("stack_trace", "No stack trace available")

                # Add stack trace to the recorded event
                if "args" not in recorded_event:
                    recorded_event["args"] = {}
                recorded_event["args"]["stack_trace"] = stack_trace

                # Add stack trace to corresponding aten ops
                for aten_op in corresponding_aten_ops:
                    if "args" not in aten_op:
                        aten_op["args"] = {}
                    aten_op["args"]["stack_trace"] = stack_trace

                event_mapping[node_name] = {
                    "recorded_event": recorded_event,
                    "aten_operations": corresponding_aten_ops,
                    "node": node,
                    "stack_trace": stack_trace
                }

    return event_mapping

# Apply the mapping
event_mapping = map_recorded_events_to_aten_ops_with_stack_trace(gm, trace_events)
# Print the mapping
print("\nEvent mapping results:")
for node_name, mapping in event_mapping.items():
    print(f"\nNode: {node_name}")
    print(f"  Recorded event: {mapping['recorded_event']['name']} (dur: {mapping['recorded_event']['dur']:.3f}μs)")
    print(f"  Aten operations: {[op['name'] for op in mapping['aten_operations']]}")
    if mapping['stack_trace']:
        # Show first few lines of stack trace
        stack_lines = mapping['stack_trace'].split('\n')[:3]
        print(f"  Stack trace preview: {' | '.join(stack_lines)}")

# Save modified trace with stack traces
with open("trace.json", 'w', encoding='utf-8') as f:
    json.dump(traced_data, f, indent=2, ensure_ascii=False)

print(f"\nSaved modified trace with stack traces to trace_with_stack_traces.json")
print(f"Found {len(event_mapping)} recorded events mapped to aten operations")

# # 3) Interpreter
# torch.fx.Interpreter(gm).run(inp)
