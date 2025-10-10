from __future__ import annotations

import datetime
import inspect
import os
import time
import uuid
from datetime import timezone
from typing import Any
from warnings import warn


# boto3 is an optional dependency. If it's not installed,
# we'll just not emit the metrics.
# Keeping this logic here so that callers don't have to
# worry about it.
EMIT_METRICS = False
try:
    from tools.stats.upload_stats_lib import upload_to_s3

    EMIT_METRICS = True
except ImportError as e:
    print(f"Unable to import boto3. Will not be emitting metrics.... Reason: {e}")


class EnvVarMetric:
    name: str
    env_var: str
    required: bool = True
    # Used to cast the value of the env_var to the correct type (defaults to str)
    type_conversion_fn: Any = None

    def __init__(
        self,
        name: str,
        env_var: str,
        required: bool = True,
        type_conversion_fn: Any = None,
    ) -> None:
        self.name = name
        self.env_var = env_var
        self.required = required
        self.type_conversion_fn = type_conversion_fn

    def value(self) -> Any:
        value = os.environ.get(self.env_var)

        # Github CI will set some env vars to an empty string
        DEFAULT_ENVVAR_VALUES = [None, ""]
        if value in DEFAULT_ENVVAR_VALUES:
            if not self.required:
                return None

            raise ValueError(
                f"Missing {self.name}. Please set the {self.env_var} "
                "environment variable to pass in this value."
            )

        if self.type_conversion_fn:
            return self.type_conversion_fn(value)
        return value


global_metrics: dict[str, Any] = {}


def add_global_metric(metric_name: str, metric_value: Any) -> None:
    """
    Adds stats that should be emitted with every metric by the current process.
    If the emit_metrics method specifies a metric with the same name, it will
    overwrite this value.
    """
    global_metrics[metric_name] = metric_value


def emit_metric(
    metric_name: str,
    metrics: dict[str, Any],
) -> None:
    """
    Upload a metric to DynamoDB (and from there, the HUD backend database).

    Even if EMIT_METRICS is set to False, this function will still run the code to
    validate and shape the metrics, skipping just the upload.

    Parameters:
        metric_name:
            Name of the metric. Every unique metric should have a different name
            and be emitted just once per run attempt.
            Metrics are namespaced by their module and the function that emitted them.
        metrics: The actual data to record.

    Some default values are populated from environment variables, which must be set
    for metrics to be emitted. (If they're not set, this function becomes a noop):
    """

    if metrics is None:
        raise ValueError("You didn't ask to upload any metrics!")

    # Merge the given metrics with the global metrics, overwriting any duplicates
    # with the given metrics.
    metrics = {**global_metrics, **metrics}

    # We use these env vars that to determine basic info about the workflow run.
    # By using env vars, we don't have to pass this info around to every function.
    # It also helps ensure that we only emit metrics during CI
    env_var_metrics = [
        EnvVarMetric("repo", "GITHUB_REPOSITORY"),
        EnvVarMetric("workflow", "GITHUB_WORKFLOW"),
        EnvVarMetric("build_environment", "BUILD_ENVIRONMENT", required=False),
        EnvVarMetric("job", "GITHUB_JOB"),
        EnvVarMetric("test_config", "TEST_CONFIG", required=False),
        EnvVarMetric("pr_number", "PR_NUMBER", required=False, type_conversion_fn=int),
        EnvVarMetric("run_id", "GITHUB_RUN_ID", type_conversion_fn=int),
        EnvVarMetric("run_number", "GITHUB_RUN_NUMBER", type_conversion_fn=int),
        EnvVarMetric("run_attempt", "GITHUB_RUN_ATTEMPT", type_conversion_fn=int),
        EnvVarMetric("job_id", "JOB_ID", type_conversion_fn=int),
        EnvVarMetric("job_name", "JOB_NAME"),
    ]

    # Use info about the function that invoked this one as a namespace and a way to filter metrics.
    calling_frame = inspect.currentframe().f_back  # type: ignore[union-attr]
    calling_frame_info = inspect.getframeinfo(calling_frame)  # type: ignore[arg-type]
    calling_file = os.path.basename(calling_frame_info.filename)
    calling_module = inspect.getmodule(calling_frame).__name__  # type: ignore[union-attr]
    calling_function = calling_frame_info.function

    try:
        default_metrics = {
            "metric_name": metric_name,
            "calling_file": calling_file,
            "calling_module": calling_module,
            "calling_function": calling_function,
            "timestamp": datetime.datetime.now(timezone.utc).strftime(
                "%Y-%m-%d %H:%M:%S.%f"
            ),
            **{m.name: m.value() for m in env_var_metrics if m.value()},
        }
    except ValueError as e:
        warn(f"Not emitting metrics for {metric_name}. {e}")
        return

    # Prefix key with metric name and timestamp to derisk chance of a uuid1 name collision
    s3_key = f"{metric_name}_{int(time.time())}_{uuid.uuid1().hex}"

    if EMIT_METRICS:
        try:
            upload_to_s3(
                bucket_name="ossci-raw-job-status",
                key=f"ossci_uploaded_metrics/{s3_key}",
                docs=[{**default_metrics, "info": metrics}],
            )
        except Exception as e:
            # We don't want to fail the job if we can't upload the metric.
            # We still raise the ValueErrors outside this try block since those indicate improperly configured metrics
            warn(f"Error uploading metric {metric_name} to DynamoDB: {e}")
            return
    else:
        print(f"Not emitting metrics for {metric_name}. Boto wasn't imported.")
