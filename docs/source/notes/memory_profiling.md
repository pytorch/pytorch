(memory-profiling)=
# Memory Profiling with Memory Snapshots

Understanding and optimizing memory usage is critical for deep learning
workloads. This note covers the modern approach to memory profiling in PyTorch
using **memory snapshots**, which provide a granular view of the memory
allocator's state.

```{note}
The ``export_memory_timeline`` function is deprecated and has known issues
with eager mode models. The memory snapshot approach described here is the
recommended replacement.
```

## Capturing a Memory Snapshot

A memory snapshot captures detailed information about every tensor currently
live in memory, including its size, shape, and the Python call stack at the
time of its allocation.

```python
import torch

# Enable memory history recording
torch.cuda.memory._record_memory_history(max_entries=100000)

# ... your model code ...

# Capture a snapshot
snapshot = torch.cuda.memory._snapshot()

# Save the snapshot for analysis
import pickle
with open("memory_snapshot.pkl", "wb") as f:
    pickle.dump(snapshot, f)

# Stop recording
torch.cuda.memory._record_memory_history(enabled=None)
```

## Analyzing Memory Snapshots

The captured snapshot can be visualized using the PyTorch memory visualization
tool at `pytorch.org/memory_viz <https://pytorch.org/memory_viz>`_. Upload the
snapshot file to get an interactive visualization of memory usage over time.

The visualization shows memory allocations and deallocations as a timeline,
categorized by tensor type (parameters, gradients, activations). This makes it
easy to identify which parts of your model use the most memory and where memory
is held longer than expected.

## Practical Memory Profiling Workflow

A systematic approach to memory profiling involves four steps:

1. **Establish a baseline** by capturing a snapshot at the start and end of a
   training iteration to understand steady-state memory usage.
2. **Identify memory growth** by comparing snapshots across multiple iterations.
   A steady increase over time is a strong indicator of a memory leak.
3. **Pinpoint the source** by using the call stack information in the snapshot
   to identify the exact line of code responsible.
4. **Optimize and verify** by applying a fix and re-capturing snapshots to
   confirm the issue is resolved.

## Combining with the Profiler

Memory snapshots complement ``torch.profiler`` traces. While the profiler shows
*when* memory allocations happen relative to operators on a timeline, memory
snapshots show *what* is allocated and *where* in the code each allocation
originates. A typical workflow is:

1. Use ``torch.profiler.profile(profile_memory=True)`` to identify which
   training step or operator causes memory spikes.
2. Use ``torch.cuda.memory._record_memory_history()`` around the suspicious
   code to capture a detailed snapshot.
3. Analyze the snapshot at `pytorch.org/memory_viz <https://pytorch.org/memory_viz>`_
   to find the root cause.

```{seealso}
- {doc}`/profiler` — ``torch.profiler`` API reference and usage guide
- {doc}`/torch_cuda_memory` — CUDA memory management APIs
```
