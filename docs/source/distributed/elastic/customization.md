# Customization

This section describes how to customize TorchElastic to fit your needs.

## Launcher

The launcher program that ships with TorchElastic
should be sufficient for most use-cases (see {ref}`launcher-api`).
You can implement a custom launcher by
programmatically creating an agent and passing it specs for your workers as
shown below.

```python
# my_launcher.py

import uuid

from torch.distributed.elastic.agent.server import WorkerSpec
from torch.distributed.elastic.agent.server.local_elastic_agent import LocalElasticAgent
from torch.distributed.elastic.multiprocessing import DefaultLogsSpecs
from torch.distributed.elastic.rendezvous import RendezvousParameters
from torch.distributed.elastic.rendezvous.registry import get_rendezvous_handler


def trainer(x):
    return x * 2


if __name__ == "__main__":
    rdzv_handler = get_rendezvous_handler(
        RendezvousParameters(
            backend="c10d",
            endpoint="localhost:0",
            run_id=str(uuid.uuid4()),
            min_nodes=1,
            max_nodes=1,
        )
    )
    spec = WorkerSpec(
        role="trainer",
        local_world_size=2,
        entrypoint=trainer,
        args=(21,),
        rdzv_handler=rdzv_handler,
        max_restarts=3,
        monitor_interval=1,
    )

    agent = LocalElasticAgent(spec, logs_specs=DefaultLogsSpecs(), start_method="spawn")
    try:
        run_result = agent.run()
        if run_result.is_failed():
            print(f"worker 0 failed with: {run_result.failures[0]}")
        else:
            print(f"worker 0 return value is: {run_result.return_values[0]}")
    finally:
        rdzv_handler.shutdown()
```

## Rendezvous Handler

To implement your own rendezvous, extend `torch.distributed.elastic.rendezvous.RendezvousHandler`
and implement its methods.

```{warning}
Rendezvous handlers are tricky to implement. Before you begin
make sure you completely understand the properties of rendezvous.
Please refer to {ref}`rendezvous-api` for more information.
```

Once implemented you can pass your custom rendezvous handler to the worker
spec when creating the agent.

```python
spec = WorkerSpec(
    role="trainer",
    rdzv_handler=MyRendezvousHandler(params),
    ...
)
elastic_agent = LocalElasticAgent(spec, logs_specs=DefaultLogsSpecs())
elastic_agent.run(spec.role)
```

To create it by name with `get_rendezvous_handler`, register a `creator`
that takes a `RendezvousParameters` and returns the handler with
`torch.distributed.elastic.rendezvous.rendezvous_handler_registry.register(name, creator)`.
The handler's `get_backend()` must return `name`.

`torchrun` runs in its own process, so to make the backend available to
`torchrun --rdzv-backend=<name>`, expose it through the `torchrun.handlers`
entry-point group of an installed package. The entry point is a zero-argument
function that returns the `creator`:

```toml
# pyproject.toml
[project.entry-points."torchrun.handlers"]
my_backend = "my_package.rendezvous:get_creator"
```

## Metric Handler

TorchElastic emits platform level metrics (see {ref}`metrics-api`).
By default metrics are emitted to `/dev/null` so you will not see them.
To have the metrics pushed to a metric handling service in your infrastructure,
implement a `torch.distributed.elastic.metrics.MetricHandler` and `configure` it in your
custom launcher.

```python
# my_launcher.py

import torch.distributed.elastic.metrics as metrics


class MyMetricHandler(metrics.MetricHandler):
    def emit(self, metric_data: metrics.MetricData):
        ...  # push metric_data to your metric sink


def main():
    metrics.configure(MyMetricHandler(), group="torchelastic")

    spec = WorkerSpec(...)
    agent = LocalElasticAgent(spec, logs_specs=DefaultLogsSpecs())
    agent.run()
```

## Events Handler

TorchElastic records events (see {ref}`events-api`) through Python `logging`.
The destination is chosen by name with `torchrun --event-log-handler` or
`WorkerSpec.event_log_handler`; `null` (default) drops events and `console`
prints them. Each event is logged as a JSON string to the logger
`torchelastic-events-<destination>`, so you can forward events elsewhere
(e.g. AWS CloudWatch) by attaching a `logging.Handler` to that logger in
your custom launcher. Rendezvous events (`RdzvEvent`) ignore the destination
and always go to the `torchelastic-events-dynamic_rendezvous` logger. They
are not recorded at all while the `dynamic_rendezvous` entry in
`torch.distributed.elastic.events.handlers` is a `logging.NullHandler` (the
default); attaching a handler to the logger alone is not enough.

```python
# my_launcher.py

import logging

from torch.distributed.elastic.events import Event


class MyEventHandler(logging.Handler):
    def emit(self, record: logging.LogRecord):
        event = Event.deserialize(record.getMessage())
        ...  # process event


def main():
    logging.getLogger("torchelastic-events-console").addHandler(MyEventHandler())

    spec = WorkerSpec(..., event_log_handler="console")
    agent = LocalElasticAgent(spec, logs_specs=DefaultLogsSpecs())
    agent.run()
```
