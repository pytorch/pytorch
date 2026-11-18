# Elastic Agent

```{eval-rst}
.. automodule:: torch.distributed.elastic.agent
```

```{eval-rst}
.. currentmodule:: torch.distributed.elastic.agent
```

## Server

```{eval-rst}
.. automodule:: torch.distributed.elastic.agent.server
```

Below is a diagram of an agent that manages a local group of workers.

```{image} agent_diagram.jpg
```

## Concepts

This section describes the high-level classes and concepts that
are relevant to understanding the role of the `agent` in torchelastic.

```{eval-rst}
.. currentmodule:: torch.distributed.elastic.agent.server
```

```{eval-rst}
.. autoclass:: ElasticAgent
   :members:
```

```{eval-rst}
.. autoclass:: WorkerSpec
   :members:
```

```{eval-rst}
.. autoclass:: WorkerState
   :members:
```

```{eval-rst}
.. autoclass:: Worker
   :members:
```

```{eval-rst}
.. autoclass:: WorkerGroup
   :members:
```

## Implementations

Below are the agent implementations provided by torchelastic.

```{eval-rst}
.. currentmodule:: torch.distributed.elastic.agent.server.local_elastic_agent
```

```{eval-rst}
.. autoclass:: LocalElasticAgent

```

## Extending the Agent

To extend the agent you can implement `ElasticAgent` directly, however
we recommend you extend `SimpleElasticAgent` instead, which provides
most of the scaffolding and leaves you with a few specific abstract methods
to implement.

```{eval-rst}
.. currentmodule:: torch.distributed.elastic.agent.server
```

```{eval-rst}
.. autoclass:: SimpleElasticAgent
   :members:
   :private-members:
```

```{eval-rst}
.. autoclass:: torch.distributed.elastic.agent.server.api.RunResult

```

## Watchdog in the Agent

A named pipe based watchdog can be enabled in `LocalElasticAgent` if an
environment variable `TORCHELASTIC_ENABLE_FILE_TIMER` with value 1 has
been defined in the `LocalElasticAgent` process.
Optionally, another environment variable `TORCHELASTIC_TIMER_FILE`
can be set with a unique file name for the named pipe. If the environment
variable `TORCHELASTIC_TIMER_FILE` is not set, `LocalElasticAgent`
will internally create a unique file name and set it to the environment
variable `TORCHELASTIC_TIMER_FILE`, and this environment variable will
be propagated to the worker processes to allow them to connect to the same
named pipe that `LocalElasticAgent` uses.

## Health Check Server

A health check monitoring server can be enabled in `LocalElasticAgent`
if an environment variable `TORCHELASTIC_HEALTH_CHECK_PORT` has been defined
in the `LocalElasticAgent` process.
Adding interface for health check server which can be extended by starting tcp/http
server on the specified port number.
Additionally, health check server will have callback to check watchdog is alive.

```{eval-rst}
.. automodule:: torch.distributed.elastic.agent.server.health_check_server
```

```{eval-rst}
.. autoclass:: HealthCheckServer
   :members:
```

```{eval-rst}
.. autofunction:: torch.distributed.elastic.agent.server.health_check_server.create_healthcheck_server
```
