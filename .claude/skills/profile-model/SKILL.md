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

### Step 1: Get Trace Path

Ask the user for the trace file path. Traces are typically:
- Single file: `/path/to/trace.json` or `/path/to/trace.json.gz`
- Directory (multi-rank): `/path/to/traces/` containing `rank0.json`, `rank1.json`, etc.

### Step 2: Load Trace as DataFrame

Load the trace into a pandas DataFrame for flexible querying. **Ask the user first:**

> "Would you like me to cache this trace as Parquet for faster future loads?
> This saves to `/tmp/trace_cache/` and makes subsequent loads faster.
> (Recommended for large traces or if you plan to ask follow-up questions)"

```python
import pandas as pd
import json
import gzip
import os

def load_trace_as_dataframe(path: str) -> pd.DataFrame:
    """
    Load a trace file into a pandas DataFrame.

    Columns:
    - name, cat, ph, ts, dur, pid, tid (standard trace fields)
    - args_* columns (flattened from args dict)
    """
    if path.endswith('.gz'):
        with gzip.open(path, 'rt') as f:
            trace = json.load(f)
    else:
        with open(path, 'r') as f:
            trace = json.load(f)

    events = trace.get("traceEvents", [])

    records = []
    for e in events:
        record = {
            "name": e.get("name"),
            "cat": e.get("cat"),
            "ph": e.get("ph"),
            "ts": e.get("ts"),
            "dur": e.get("dur"),
            "pid": e.get("pid"),
            "tid": e.get("tid"),
            "id": e.get("id"),
        }

        # Flatten args dict into separate columns
        args = e.get("args", {})
        for key, value in args.items():
            if isinstance(value, (list, dict)):
                value = str(value)
            record[f"args_{key}"] = value

        records.append(record)

    df = pd.DataFrame(records)
    df["ts"] = pd.to_numeric(df["ts"], errors="coerce")
    df["dur"] = pd.to_numeric(df["dur"], errors="coerce")

    return df


def load_trace_with_cache(path: str, use_cache: bool = True) -> pd.DataFrame:
    """
    Load trace with optional Parquet caching for faster subsequent loads.

    Parquet loading is faster than JSON parsing.

    Note: Trace args columns often have mixed types (int, str, None, etc.).
    This function handles mixed types gracefully when saving to Parquet.
    """
    cache_dir = "/tmp/trace_cache"
    basename = os.path.basename(path).replace(".json.gz", "").replace(".json", "")
    cache_path = f"{cache_dir}/{basename}.parquet"

    # Check for existing cache
    if use_cache and os.path.exists(cache_path):
        cache_mtime = os.path.getmtime(cache_path)
        source_mtime = os.path.getmtime(path)
        if cache_mtime > source_mtime:
            print(f"✓ Loading from cache: {cache_path}")
            return pd.read_parquet(cache_path)

    # Load from JSON
    print(f"Parsing trace: {path}")
    file_size_mb = os.path.getsize(path) / (1024 * 1024)
    if file_size_mb > 100:
        print(f"  Large file ({file_size_mb:.0f}MB) - this may take a moment...")

    df = load_trace_as_dataframe(path)
    print(f"  Loaded {len(df):,} events")

    # Save to cache if requested
    if use_cache:
        os.makedirs(cache_dir, exist_ok=True)

        # Handle mixed types in args_* columns before saving to Parquet
        # Parquet requires homogeneous types per column
        df_to_save = df.copy()
        for col in df_to_save.columns:
            if col.startswith("args_"):
                # Check if column has mixed types
                non_null = df_to_save[col].dropna()
                if len(non_null) > 0:
                    types = non_null.apply(type).unique()
                    if len(types) > 1:
                        # Mixed types detected - convert entire column to string
                        df_to_save[col] = df_to_save[col].astype(str)
                        # Replace 'nan' strings back to None for cleaner data
                        df_to_save[col] = df_to_save[col].replace({'nan': None, 'None': None})

        try:
            df_to_save.to_parquet(cache_path)
            print(f"✓ Cached to: {cache_path}")
            print(f"  (Future loads will be faster)")
        except Exception as e:
            # If Parquet save still fails, fall back to converting all args to strings
            print(f"  Warning: Parquet save failed ({e}), retrying with string conversion...")
            for col in df_to_save.columns:
                if col.startswith("args_"):
                    df_to_save[col] = df_to_save[col].astype(str).replace({'nan': None, 'None': None})
            try:
                df_to_save.to_parquet(cache_path)
                print(f"✓ Cached to: {cache_path}")
            except Exception as e2:
                print(f"  ⚠️ Could not cache to Parquet: {e2}")
                print(f"  Continuing without cache...")

    return df


# Load the trace (set use_cache based on user preference)
df = load_trace_with_cache("/path/to/trace.json.gz", use_cache=True)
```

### Step 3: Quick Summary

After loading, show a quick summary:

```python
def print_trace_summary(df: pd.DataFrame) -> None:
    """Print a quick summary of the trace."""
    print("=" * 60)
    print("TRACE SUMMARY")
    print("=" * 60)
    print(f"Total events: {len(df):,}")
    print(f"\nEvents by category:")
    print(df["cat"].value_counts().to_string())

    # GPU utilization
    kernels = df[df["cat"] == "kernel"]
    if len(kernels) > 0:
        total_kernel_time = kernels["dur"].sum()

        trace_start = df["ts"].min()
        trace_end = (df["ts"] + df["dur"].fillna(0)).max()
        trace_duration = trace_end - trace_start

        gpu_util = 100 * total_kernel_time / trace_duration if trace_duration > 0 else 0

        print(f"\nGPU Metrics:")
        print(f"  Kernel count: {len(kernels):,}")
        print(f"  Total kernel time: {total_kernel_time/1e6:.2f}s")
        print(f"  Trace duration: {trace_duration/1e6:.2f}s")
        print(f"  GPU utilization: {gpu_util:.1f}%")

        if gpu_util < 50:
            print(f"  ⚠️  Low GPU utilization - GPU idle >50% of time")

print_trace_summary(df)
```

### Step 4: Analyze with DataFrame Queries

The DataFrame is now loaded. Use pandas queries to answer user questions:

```python
# Top 10 GPU kernels by total time
kernels = df[df["cat"] == "kernel"]
top_kernels = (
    kernels.groupby("name")["dur"]
    .agg(total_us="sum", count="count", avg_us="mean")
    .sort_values("total_us", ascending=False)
    .head(10)
)
total_kernel_time = kernels["dur"].sum()
top_kernels["pct"] = 100 * top_kernels["total_us"] / total_kernel_time

print("=" * 60)
print("TOP 10 GPU KERNELS BY TIME")
print("=" * 60)
print(top_kernels[["pct", "count", "avg_us"]].round(1).to_string())
```

#### Common DataFrame Queries

| User Question | Pandas Query |
|---------------|--------------|
| "Top kernels by time" | `df[df["cat"]=="kernel"].groupby("name")["dur"].sum().sort_values(ascending=False).head(10)` |
| "Show NCCL ops" | `df[df["name"].str.contains("nccl", case=False, na=False)]` |
| "Memory operations" | `df[df["cat"].str.contains("mem", case=False, na=False)]` |
| "Slow events (>1ms)" | `df[df["dur"] > 1000].sort_values("dur", ascending=False)` |
| "Events on stream 7" | `df[df["args_stream"] == 7]` |
| "CPU ops only" | `df[df["cat"] == "cpu_op"]` |
| "Time window 1-2s" | `df[(df["ts"] > 1e6) & (df["ts"] < 2e6)]` |
| "Unique kernel names" | `df[df["cat"]=="kernel"]["name"].unique()` |
| "Group by thread" | `df.groupby("tid")["dur"].sum().sort_values(ascending=False)` |

### Step 5: GPU Utilization & Idle Gap Analysis

```python
def analyze_gpu_utilization(df: pd.DataFrame) -> dict:
    """Calculate GPU utilization metrics."""
    kernels = df[(df["cat"] == "kernel") & df["dur"].notna() & df["ts"].notna()]

    if len(kernels) == 0:
        return {"error": "No GPU kernels found"}

    trace_start = df["ts"].min()
    trace_end = (df["ts"] + df["dur"].fillna(0)).max()
    trace_duration = trace_end - trace_start
    total_kernel_time = kernels["dur"].sum()

    return {
        "gpu_utilization_pct": round(100 * total_kernel_time / trace_duration, 1),
        "trace_duration_s": round(trace_duration / 1e6, 3),
        "total_kernel_time_s": round(total_kernel_time / 1e6, 3),
        "kernel_count": len(kernels),
    }


def find_idle_gaps(df: pd.DataFrame, min_gap_us: float = 1000) -> pd.DataFrame:
    """Find idle gaps between GPU kernels (default: gaps > 1ms)."""
    kernels = (
        df[(df["cat"] == "kernel") & df["dur"].notna() & df["ts"].notna()]
        .sort_values("ts")
        .copy()
    )

    if len(kernels) < 2:
        return pd.DataFrame()

    # Calculate gaps between consecutive kernels
    kernels["end_ts"] = kernels["ts"] + kernels["dur"]
    kernels["next_start"] = kernels["ts"].shift(-1)
    kernels["gap"] = kernels["next_start"] - kernels["end_ts"]

    # Filter to significant gaps
    gaps = kernels[kernels["gap"] >= min_gap_us][["name", "gap", "end_ts"]].copy()
    gaps.columns = ["after_kernel", "gap_us", "timestamp"]
    gaps["gap_ms"] = (gaps["gap_us"] / 1000).round(2)

    return gaps.sort_values("gap_us", ascending=False).head(20)


# Run analysis
util = analyze_gpu_utilization(df)
print(f"\nGPU Utilization: {util['gpu_utilization_pct']}%")

if util['gpu_utilization_pct'] < 30:
    print("🔴 SEVERE: GPU utilization below 30% - likely CPU-bound")
elif util['gpu_utilization_pct'] < 50:
    print("🟡 WARNING: GPU utilization below 50%")
else:
    print("🟢 GPU utilization looks healthy")

gaps = find_idle_gaps(df, min_gap_us=1000)
if len(gaps) > 0:
    print(f"\nTop idle gaps (>1ms):")
    print(gaps[["gap_ms", "after_kernel"]].head(10).to_string())

    if gaps["gap_ms"].max() > 10:
        print(f"\n🔴 Found large gaps >10ms - investigate CPU bottlenecks")
```

### Step 6: Interpret Results

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
except ImportError:
    print("HTA not installed. Install with: pip install HolisticTraceAnalysis")
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
