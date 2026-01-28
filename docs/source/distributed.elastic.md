# Torch Distributed Elastic

Torch Distributed Elastic (also known as PyTorch Elastic) is a framework that enables fault-tolerant and elastic distributed training for PyTorch applications. It extends PyTorch's distributed training capabilities to handle real-world production scenarios where worker nodes may fail, be preempted, or need to be dynamically scaled.

## What is PyTorch Elastic?

PyTorch Elastic provides two key capabilities that make distributed training more robust and flexible:

### Fault Tolerance

PyTorch Elastic automatically handles worker failures during training. When a worker process crashes, encounters an error, or is terminated, the framework:

- Detects the failure and coordinates with other workers
- Automatically restarts failed workers and re-establishes the distributed training group
- Resumes training from the last checkpoint (if your training script supports checkpointing)
- Configures a maximum number of restart attempts to prevent infinite retry loops

This is particularly valuable for long-running training jobs where hardware failures, network issues, or other transient errors could otherwise cause the entire training run to fail.

### Elasticity

PyTorch Elastic supports dynamic scaling of the number of workers during training. You can specify a minimum and maximum number of nodes, and the framework will:

- Automatically incorporate new nodes as they become available (scale up)
- Handle node removal gracefully (scale down)
- Re-rendezvous the worker group when membership changes
- Maintain training continuity across membership changes

This elasticity is especially useful in cloud environments where nodes may be preempted, or when you want to take advantage of additional compute resources as they become available.

## When to Use PyTorch Elastic

Consider using PyTorch Elastic when:

- Running long-running training jobs that need to survive worker failures
- Training in cloud environments where nodes may be preempted or terminated
- You want to dynamically scale your training cluster based on resource availability
- You need production-grade reliability for distributed training workloads

For simple single-node or multi-node training without fault tolerance requirements, you can use the standard `torch.distributed` APIs or `torchrun` without elasticity features.

## Get Started

```{toctree}
:caption: Usage
:maxdepth: 1

elastic/quickstart
elastic/train_script
elastic/examples
```

## Documentation

```{toctree}
:caption: API
:maxdepth: 1

elastic/run
elastic/agent
elastic/multiprocessing
elastic/errors
elastic/rendezvous
elastic/timer
elastic/metrics
elastic/events
elastic/subprocess_handler
elastic/control_plane
elastic/numa
```

```{toctree}
:caption: Advanced
:maxdepth: 1

elastic/customization
```

```{toctree}
:caption: Plugins
:maxdepth: 1

elastic/kubernetes
```
