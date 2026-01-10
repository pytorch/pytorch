# Owner(s): ["module: onnx"]
"""Error reproduction utilities for op consistency tests."""

from __future__ import annotations

import difflib
import pathlib
import platform
import sys
import time
import traceback
from typing import Any, TYPE_CHECKING

import numpy as np
import onnx
import onnxruntime as ort
import onnxscript

import torch


if TYPE_CHECKING:
    from collections.abc import Mapping


_REPRODUCTION_TEMPLATE = '''\
import google.protobuf.text_format
import numpy as np
from numpy import array, float16, float32, float64, int32, int64
import onnx
import onnxruntime as ort

# Run n times
N = 1

onnx_model_text = """
{onnx_model_text}
"""

ort_inputs = {ort_inputs}

# Set up the inference session
session_options = ort.SessionOptions()
session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
onnx_model = onnx.ModelProto()
google.protobuf.text_format.Parse(onnx_model_text, onnx_model)

# Uncomment this line to save the model to a file for examination
# onnx.save_model(onnx_model, "{short_test_name}.onnx")

onnx.checker.check_model(onnx_model)
session = ort.InferenceSession(onnx_model.SerializeToString(), session_options, providers=("CPUExecutionProvider",))

# Run the model
for _ in range(N):
    ort_outputs = session.run(None, ort_inputs)
'''

_ISSUE_MARKDOWN_TEMPLATE = """
### Summary

ONNX Runtime raises `{error_text}` when executing test `{test_name}` in ONNX Script `TorchLib`.

To recreate this report, use

```bash
CREATE_REPRODUCTION_REPORT=1 python -m pytest {test_file_path} -k {short_test_name}
```

### To reproduce

```python
{reproduction_code}
```

### Full error stack

```
{error_stack}
```

### The ONNX model text for visualization

```
{onnx_model_textual_representation}
```

### Environment

```
{sys_info}
```
"""


_MISMATCH_MARKDOWN_TEMPLATE = """\
### Summary

The output of ONNX Runtime does not match that of PyTorch when executing test
`{test_name}`, `sample {sample_num}` in ONNX Script `TorchLib`.

To recreate this report, use

```bash
CREATE_REPRODUCTION_REPORT=1 python -m pytest {test_file_path} -k {short_test_name}
```

### Inputs

Shapes: `{input_shapes}`

<details><summary>Details</summary>
<p>

```python
kwargs = {kwargs}
inputs = {inputs}
```

</p>
</details>

### Expected output

Shape: `{expected_shape}`

<details><summary>Details</summary>
<p>

```python
expected = {expected}
```

</p>
</details>

### Actual output

Shape: `{actual_shape}`

<details><summary>Details</summary>
<p>

```python
actual = {actual}
```

</p>
</details>

### Difference

<details><summary>Details</summary>
<p>

```diff
{diff}
```

</p>
</details>

### ONNX Model

```
{onnx_model_text}
```

### Full error stack

```
{error_stack}
```

### Environment

```
{sys_info}
```

"""


def create_reproduction_report(
    test_name: str,
    onnx_model: onnx.ModelProto,
    ort_inputs: Mapping[str, Any],
    error: Exception,
    test_file_path: str,
) -> None:
    # NOTE: We choose to embed the ONNX model as a string in the report instead of
    # saving it to a file because it is easier to share the report with others.
    onnx_model_text = str(onnx_model)
    with np.printoptions(threshold=sys.maxsize):
        ort_inputs = dict(ort_inputs.items())
        input_text = str(ort_inputs)
    error_text = str(error)
    error_stack = error_text + "\n" + "".join(traceback.format_tb(error.__traceback__))
    sys_info = f"""\
OS: {platform.platform()}
Python version: {sys.version}
onnx=={onnx.__version__}
onnxruntime=={ort.__version__}
onnxscript=={onnxscript.__version__}
numpy=={np.__version__}
torch=={torch.__version__}"""
    short_test_name = test_name.rsplit(".", maxsplit=1)[-1]
    reproduction_code = _REPRODUCTION_TEMPLATE.format(
        onnx_model_text=onnx_model_text,
        ort_inputs=input_text,
        short_test_name=short_test_name,
    )
    onnx_model_textual_representation = onnx.printer.to_text(onnx_model)

    markdown = _ISSUE_MARKDOWN_TEMPLATE.format(
        error_text=error_text,
        test_name=test_name,
        short_test_name=short_test_name,
        reproduction_code=reproduction_code,
        error_stack=error_stack,
        sys_info=sys_info,
        onnx_model_textual_representation=onnx_model_textual_representation,
        test_file_path=test_file_path,
    )

    # Turn test name into a valid file name
    markdown_file_name = f"{short_test_name.replace('/', '-').replace(':', '-')}-{str(time.time()).replace('.', '_')}.md"
    markdown_file_path = save_error_report(markdown_file_name, markdown)
    print(f"Created reproduction report at {markdown_file_path}")


def create_mismatch_report(
    test_name: str,
    sample_num: int,
    onnx_model: onnx.ModelProto,
    inputs,
    kwargs,
    actual,
    expected,
    error: Exception,
    test_file_path: str,
) -> None:
    torch.set_printoptions(threshold=sys.maxsize)

    error_text = str(error)
    error_stack = error_text + "\n" + "".join(traceback.format_tb(error.__traceback__))
    short_test_name = test_name.rsplit(".", maxsplit=1)[-1]
    diff = difflib.unified_diff(
        str(actual).splitlines(),
        str(expected).splitlines(),
        fromfile="actual",
        tofile="expected",
        lineterm="",
    )
    onnx_model_text = onnx.printer.to_text(onnx_model)
    input_shapes = repr(
        [
            f"Tensor<{inp.shape}, dtype={inp.dtype}>"
            if isinstance(inp, torch.Tensor)
            else inp
            for inp in inputs
        ]
    )
    sys_info = f"""\
OS: {platform.platform()}
Python version: {sys.version}
onnx=={onnx.__version__}
onnxruntime=={ort.__version__}
onnxscript=={onnxscript.__version__}
numpy=={np.__version__}
torch=={torch.__version__}"""
    markdown = _MISMATCH_MARKDOWN_TEMPLATE.format(
        test_name=test_name,
        short_test_name=short_test_name,
        sample_num=sample_num,
        input_shapes=input_shapes,
        inputs=inputs,
        kwargs=kwargs,
        expected=expected,
        expected_shape=expected.shape if isinstance(expected, torch.Tensor) else None,
        actual=actual,
        actual_shape=actual.shape if isinstance(actual, torch.Tensor) else None,
        diff="\n".join(diff),
        error_stack=error_stack,
        sys_info=sys_info,
        onnx_model_text=onnx_model_text,
        test_file_path=test_file_path,
    )

    markdown_file_name = f"mismatch-{short_test_name.replace('/', '-').replace(':', '-')}-{str(time.time()).replace('.', '_')}.md"
    markdown_file_path = save_error_report(markdown_file_name, markdown)
    print(f"Created reproduction report at {markdown_file_path}")


def save_error_report(file_name: str, text: str):
    reports_dir = pathlib.Path("error_reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    file_path = reports_dir / file_name
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(text)

    return file_path
