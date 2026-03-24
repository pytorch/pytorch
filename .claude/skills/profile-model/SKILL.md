---
name: profile-model
description: Profile PyTorch models and analyze profiler traces. Use when the user wants to profile a training loop, analyze trace files (.json/.json.gz), diagnose GPU performance issues, find slow kernels, or identify idle time.
---

# Profile Model

Profile PyTorch models and analyze performance traces. Supports two workflows:

1. **Instrument & Profile**: Wrap user-specified code with `torch.profiler`, run, generate traces
2. **Analyze Existing Traces**: Load `.json`/`.json.gz` trace files and analyze

## Quick Start

Ask the user:
- **"Do you have an existing trace file?"** → Go to [Analyze Traces](#workflow-b-analyze-existing-traces)
- **"Do you want to generate a new trace?"** → Go to [Generate Traces](#workflow-a-instrument--profile)

---

## Workflow A: Instrument & Profile

### Step 1: Get Target Code from User

Ask the user to specify what to profile:
- File path + line range: "Profile `train.py` lines 50-100"
- Function name: "Profile the `train_step` function"
- Module: "Profile the forward pass of `MyModel`"

### Step 2: Generate Instrumented Wrapper

Create a wrapper script that profiles the specified code:

```python
import torch
from torch.profiler import profile, ProfilerActivity, schedule, tensorboard_trace_handler

# User's imports and setup here
# ...

# Profiler configuration
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=schedule(
        wait=1,      # Skip first iteration (warmup)
        warmup=1,    # Warmup iteration
        active=3,    # Profile 3 iterations
        repeat=1     # One cycle
    ),
    on_trace_ready=tensorboard_trace_handler('/tmp/profiler_traces'),
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
) as prof:
    for step in range(5):  # wait + warmup + active = 5
        # User's code here
        <TARGET_CODE>
        prof.step()

print(f"Traces saved to /tmp/profiler_traces/")
```

### Step 3: Run the Script

```bash
python /tmp/profile_wrapper.py
```

### Step 4: Proceed to Analysis

After traces are generated, proceed to [Analyze Traces](#workflow-b-analyze-existing-traces).

---

## Workflow B: Analyze Existing Traces

### Step 1: Load the Trace

Ask the user for the trace file path. Traces are typically:
- Single file: `/path/to/trace.json` or `/path/to/trace.json.gz`
- Directory (multi-rank): `/path/to/traces/` containing `rank0.json`, `rank1.json`, etc.

### Step 2: Quick Analysis (No Dependencies)

**Always start here.** This works with just Python standard libraries:

```python
import json
import gzip
from collections import defaultdict

def load_trace(path: str) -> dict:
    """Load a trace file (handles .json and .json.gz)"""
    if path.endswith('.gz'):
        with gzip.open(path, 'rt') as f:
            return json.load(f)
    else:
        with open(path, 'r') as f:
            return json.load(f)

def analyze_trace(trace: dict) -> dict:
    """Basic trace analysis without external dependencies"""
    events = trace.get("traceEvents", [])

    # Categorize events
    kernel_times = defaultdict(lambda: {"total_dur": 0, "count": 0})
    cpu_op_times = defaultdict(lambda: {"total_dur": 0, "count": 0})
    memory_events = []
    comm_events = []

    for event in events:
        if "dur" not in event or event.get("ph") != "X":
            continue

        name = event.get("name", "unknown")
        cat = event.get("cat", "")
        dur = event["dur"]  # microseconds

        if cat == "kernel":
            kernel_times[name]["total_dur"] += dur
            kernel_times[name]["count"] += 1
        elif cat == "cpu_op":
            cpu_op_times[name]["total_dur"] += dur
            cpu_op_times[name]["count"] += 1
        elif "memcpy" in cat.lower() or "memset" in cat.lower():
            memory_events.append(event)
        elif "nccl" in name.lower() or "collective" in cat.lower():
            comm_events.append(event)

    return {
        "kernel_times": dict(kernel_times),
        "cpu_op_times": dict(cpu_op_times),
        "memory_events": memory_events,
        "comm_events": comm_events,
        "total_events": len(events),
    }

# Load and analyze
trace = load_trace("/path/to/trace.json.gz")
results = analyze_trace(trace)

# Top 10 GPU kernels by time
kernels = sorted(
    results["kernel_times"].items(),
    key=lambda x: -x[1]["total_dur"]
)[:10]

total_kernel_time = sum(k[1]["total_dur"] for k in results["kernel_times"].items())

print("=" * 60)
print("TOP 10 GPU KERNELS BY TIME")
print("=" * 60)
for name, stats in kernels:
    pct = 100 * stats["total_dur"] / total_kernel_time if total_kernel_time > 0 else 0
    avg = stats["total_dur"] / stats["count"] if stats["count"] > 0 else 0
    print(f"{pct:5.1f}% | {stats['count']:5d} calls | avg {avg:8.1f}μs | {name[:60]}")

print(f"\nTotal GPU kernel time: {total_kernel_time/1e6:.2f}s")
print(f"Total memory operations: {len(results['memory_events'])}")
print(f"Total communication operations: {len(results['comm_events'])}")
```

### Step 3: Interpret Quick Analysis Results

| Finding | Likely Issue | Recommendation |
|---------|--------------|----------------|
| Many small kernels (avg < 10μs) | Kernel launch overhead | Use `torch.compile`, fuse operations |
| One kernel dominates (>50%) | Expected for compute-bound | Check if kernel is optimized |
| Many memory operations | Memory-bound | Reduce copies, use pinned memory |
| High communication time | Distributed overhead | Overlap comm with compute |

### Step 4: Deep Analysis with HTA (Optional)

If the user wants deeper analysis, use HTA (Holistic Trace Analysis):

```python
# Check if HTA is available
try:
    from hta.trace_analysis import TraceAnalysis
    HTA_AVAILABLE = True
except ImportError:
    HTA_AVAILABLE = False
    print("HTA not installed. Install with: pip install HolisticTraceAnalysis")
    print("Or use Buck target: //HolisticTraceAnalysis/hta:trace_analysis")
```

#### HTA Analysis Functions

| User Question | HTA Function | What It Returns |
|---------------|--------------|-----------------|
| "What's taking GPU time?" | `get_gpu_kernel_breakdown()` | Kernel time distribution |
| "Where's the critical path?" | `critical_path_analysis()` | Bottleneck path through execution |
| "Why is GPU idle?" | `get_idle_time_breakdown()` | Host wait vs kernel wait breakdown |
| "Comm/compute overlap?" | `get_comm_comp_overlap()` | Overlap percentage per rank |
| "Any slow ranks?" | `get_potential_stragglers()` | Straggler rank identification |
| "Overall time breakdown?" | `get_temporal_breakdown()` | Idle/compute/non-compute time |

#### HTA Usage Examples

```python
from hta.trace_analysis import TraceAnalysis

# Load traces
analyzer = TraceAnalysis(trace_dir="/path/to/traces/")
# OR for single file:
# analyzer = TraceAnalysis(trace_files={0: "/path/to/trace.json.gz"})

# GPU Kernel Breakdown
kernel_type_df, kernel_df = analyzer.get_gpu_kernel_breakdown(visualize=False)
print("Time by kernel type:")
print(kernel_type_df)

# Temporal Breakdown (idle vs compute vs non-compute)
temporal_df = analyzer.get_temporal_breakdown(visualize=False)
print("\nTemporal breakdown:")
print(temporal_df)

# Idle Time Breakdown (why is GPU idle?)
idle_df, interval_df = analyzer.get_idle_time_breakdown(
    ranks=[0],
    visualize=False,
    show_idle_interval_stats=True
)
print("\nIdle time breakdown:")
print(idle_df)

# Critical Path Analysis (for distributed training)
cp_graph, success = analyzer.critical_path_analysis(
    rank=0,
    annotation="ProfilerStep",
    instance_id=1  # Use second iteration (skip warmup)
)
if success:
    print("\nCritical path analysis succeeded")
    # Overlay on trace for visualization
    output_path = analyzer.overlay_critical_path_analysis(
        rank=0,
        critical_path_graph=cp_graph,
        output_dir="/tmp/critical_path_output"
    )
    print(f"Overlaid trace: {output_path}")

# Communication-Computation Overlap (for distributed)
overlap_df = analyzer.get_comm_comp_overlap(visualize=False)
print("\nComm-compute overlap:")
print(overlap_df)

# Straggler Detection (for distributed)
stragglers = analyzer.get_potential_stragglers(num_candidates=2)
print(f"\nPotential straggler ranks: {stragglers}")
```

---

## Common Performance Issues & Fixes

| Symptom | Diagnosis | Fix |
|---------|-----------|-----|
| Many small kernels | Kernel launch overhead | Use `torch.compile`, operator fusion |
| High idle time (host wait) | CPU bottleneck | Increase DataLoader `num_workers`, use `pin_memory=True` |
| High idle time (kernel wait) | Back-to-back kernel gaps | Fuse operations, reduce synchronization |
| Low comm/compute overlap | Blocking collectives | Use async collectives, overlap with compute |
| Stragglers detected | Load imbalance | Check data distribution, batch sizes across ranks |
| Memory copy overhead | Excessive data movement | Use in-place operations, reduce .cpu()/.cuda() calls |

---

## Output Format

When reporting analysis results:

1. **Start with a summary** - One sentence describing the main finding
2. **Show top issues** - Top 3-5 kernels/ops by time with percentages
3. **Identify bottleneck type** - Compute-bound, memory-bound, communication-bound, or CPU-bound
4. **Provide actionable recommendations** - Specific changes the user can make
5. **Offer deeper analysis** - Ask if they want HTA analysis for more detail

Example output:
```
## Trace Analysis Summary

**Main Finding**: GPU is idle 40% of the time, primarily due to CPU bottleneck (host wait).

### Top GPU Kernels (by time)
| % Time | Calls | Avg (μs) | Kernel |
|--------|-------|----------|--------|
| 35.2%  | 1000  | 450.3    | ampere_sgemm_128x64 |
| 22.1%  | 1000  | 283.5    | volta_nccl_reduce_scatter |
| 15.8%  | 2000  | 101.2    | elementwise_kernel |

### Bottleneck: CPU-Bound
The GPU is waiting for the CPU 40% of the time. This suggests data loading or preprocessing is slow.

### Recommendations
1. Increase DataLoader `num_workers` (currently appears to be 0 or 1)
2. Enable `pin_memory=True` in DataLoader
3. Move preprocessing to GPU if possible

**Want deeper analysis?** I can use HTA to get detailed idle time breakdown and critical path analysis.
```

---

## Rules

1. **Always start with quick analysis** - It requires no dependencies
2. **Ask before installing** - Don't assume HTA is available; offer to use it
3. **Be specific** - Give exact percentages, kernel names, and recommendations
4. **Offer next steps** - Always ask if the user wants deeper analysis
5. **Handle errors gracefully** - If trace format is unexpected, explain what's missing
