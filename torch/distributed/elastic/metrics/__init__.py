#!/usr/bin/env/python3
# mypy: allow-untyped-defs

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Metrics API.

**Overview**:

The metrics API in torchelastic is used to publish telemetry metrics.
It is designed to be used by torchelastic's internal modules to
publish metrics for the end user with the goal of increasing visibility
and helping with debugging. However you may use the same API in your
jobs to publish metrics to the same metrics ``sink``.

A ``metric`` can be thought of as timeseries data
and is uniquely identified by the string-valued tuple
``(metric_group, metric_name)``.

torchelastic makes no assumptions about what a ``metric_group`` is
and what relationship it has with ``metric_name``. It is totally up
to the user to use these two fields to uniquely identify a metric.

.. note:: The metric group ``torchelastic`` is reserved by torchelastic for
          platform level metrics that it produces.
          For instance torchelastic may output the latency (in milliseconds)
          of a re-rendezvous operation from the agent as
          ``(torchelastic, agent.rendezvous.duration.ms)``

A sensible way to use metric groups is to map them to a stage or module
in your job. You may also encode certain high level properties
the job such as the region or stage (dev vs prod).

**Publish Metrics**:

Using torchelastic's metrics API is similar to using python's logging
framework. You first have to configure a metrics handler before
trying to add metric data.

The example below measures the latency for the ``calculate()`` function.

::

  import time
  import torch.distributed.elastic.metrics as metrics

  # makes all metrics other than the one from "my_module" to go /dev/null
  metrics.configure(metrics.NullMetricsHandler())
  metrics.configure(metrics.ConsoleMetricsHandler(), "my_module")

  def my_method():
    start = time.time()
    calculate()
    end = time.time()
    metrics.put_metric("calculate_latency", int(end-start), "my_module")

You may also use the torch.distributed.elastic.metrics.prof` decorator
to conveniently and succinctly profile functions

::

  # -- in module examples.foobar --

  import torch.distributed.elastic.metrics as metrics

  metrics.configure(metrics.ConsoleMetricsHandler(), "foobar")
  metrics.configure(metrics.ConsoleMetricsHandler(), "Bar")

  @metrics.prof
  def foo():
    pass

  class Bar():

    @metrics.prof
    def baz():
        pass

``@metrics.prof`` will publish the following metrics
::

  <leaf_module or classname>.success - 1 if the function finished successfully
  <leaf_module or classname>.failure - 1 if the function threw an exception
  <leaf_module or classname>.duration.ms - function duration in milliseconds

**Configuring Metrics Handler**:

`torch.distributed.elastic.metrics.MetricHandler` is responsible for emitting
the added metric values to a particular destination. Metric groups can be
configured with different metric handlers.

By default torchelastic emits all metrics to ``/dev/null``.
By adding the following configuration metrics,
``torchelastic`` and ``my_app`` metric groups will be printed out to
console.

::

  import torch.distributed.elastic.metrics as metrics

  metrics.configure(metrics.ConsoleMetricHandler(), group = "torchelastic")
  metrics.configure(metrics.ConsoleMetricHandler(), group = "my_app")

**Writing a Custom Metric Handler**:

If you want your metrics to be emitted to a custom location, implement
the `torch.distributed.elastic.metrics.MetricHandler` interface
and configure your job to use your custom metric handler.

Below is a toy example that prints the metrics to ``stdout``

::

  import torch.distributed.elastic.metrics as metrics

  class StdoutMetricHandler(metrics.MetricHandler):
     def emit(self, metric_data):
         ts = metric_data.timestamp
         group = metric_data.group_name
         name = metric_data.name
         value = metric_data.value
         print(f"[{ts}][{group}]: {name}={value}")

  metrics.configure(StdoutMetricHandler(), group="my_app")

Now all metrics in the group ``my_app`` will be printed to stdout as:

::

  [1574213883.4182858][my_app]: my_metric=<value>
  [1574213940.5237644][my_app]: my_metric=<value>

"""

from typing import Optional

from .api import (  # noqa: F401
    configure,
    ConsoleMetricHandler,
    get_elapsed_time_ms,
    getStream,
    MetricData,
    MetricHandler,
    MetricsConfig,
    NullMetricHandler,
    prof,
    profile,
    publish_metric,
    put_metric,
)


def initialize_metrics(cfg: Optional[MetricsConfig] = None):
    pass


try:
    from torch.distributed.elastic.metrics.static_init import *  # type: ignore[import] # noqa: F401 F403
except ModuleNotFoundError:
    pass
