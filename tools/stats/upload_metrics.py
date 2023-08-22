import datetime
import inspect
import os
import time
import uuid

from decimal import Decimal
from typing import Any, Dict
from warnings import warn

# boto3 is an optional dependency. If it's not installed,
# we'll just not emit the metrics.
# Keeping this logic here so that callers don't have to
# worry about it.
EMIT_METRICS = False
try:
    import boto3  # type: ignore[import]

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
        if value is None and self.required:
            raise ValueError(
                f"Missing {self.name}. Please set the {self.env_var} "
                "environment variable to pass in this value."
            )
        if self.type_conversion_fn:
            return self.type_conversion_fn(value)
        return value


def emit_metric(
    metric_name: str,
    metrics: Dict[str, Any],
) -> None:
    """
    Upload a metric to DynamoDB (and from there, Rockset).

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

    # We use these env vars that to determine basic info about the workflow run.
    # By using env vars, we don't have to pass this info around to every function.
    # It also helps ensure that we only emit metrics during CI
    env_var_metrics = [
        EnvVarMetric("repo", "GITHUB_REPOSITORY"),
        EnvVarMetric("workflow", "GITHUB_WORKFLOW"),
        EnvVarMetric("build_environment", "BUILD_ENVIRONMENT"),
        EnvVarMetric("job", "GITHUB_JOB"),
        EnvVarMetric("test_config", "TEST_CONFIG", required=False),
        EnvVarMetric("run_id", "GITHUB_RUN_ID", type_conversion_fn=int),
        EnvVarMetric("run_number", "GITHUB_RUN_NUMBER", type_conversion_fn=int),
        EnvVarMetric("run_attempt", "GITHUB_RUN_ATTEMPT", type_conversion_fn=int),
    ]

    # Use info about the function that invoked this one as a namespace and a way to filter metrics.
    calling_frame = inspect.currentframe().f_back  # type: ignore[union-attr]
    calling_frame_info = inspect.getframeinfo(calling_frame)  # type: ignore[arg-type]
    calling_file = os.path.basename(calling_frame_info.filename)
    calling_module = inspect.getmodule(calling_frame).__name__  # type: ignore[union-attr]
    calling_function = calling_frame_info.function

    try:
        reserved_metrics = {
            "metric_name": metric_name,
            "calling_file": calling_file,
            "calling_module": calling_module,
            "calling_function": calling_function,
            "timestamp": datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S.%f"),
            **{m.name: m.value() for m in env_var_metrics},
        }
    except ValueError as e:
        warn(f"Not emitting metrics. {e}")
        return

    # Prefix key with metric name and timestamp to derisk chance of a uuid1 name collision
    reserved_metrics[
        "dynamo_key"
    ] = f"{metric_name}_{int(time.time())}_{uuid.uuid1().hex}"

    # Ensure the metrics dict doesn't contain any reserved keys
    for key in reserved_metrics.keys():
        used_reserved_keys = [k for k in metrics.keys() if k == key]
        if used_reserved_keys:
            raise ValueError(f"Metrics dict contains reserved keys: [{', '.join(key)}]")

    # boto3 doesn't support uploading float values to DynamoDB, so convert them all to decimals.
    metrics = _convert_float_values_to_decimals(metrics)

    if EMIT_METRICS:
        try:
            session = boto3.Session(region_name="us-east-1")
            session.resource("dynamodb").Table("torchci-metrics").put_item(
                Item={
                    **reserved_metrics,
                    **metrics,
                }
            )
        except Exception as e:
            # We don't want to fail the job if we can't upload the metric.
            # We still raise the ValueErrors outside this try block since those indicate improperly configured metrics
            warn(f"Error uploading metric to DynamoDB: {e}")
            return


def _convert_float_values_to_decimals(data: Dict[str, Any]) -> Dict[str, Any]:
    return {k: Decimal(str(v)) if isinstance(v, float) else v for k, v in data.items()}
