"""Error reproduction utilities for op consistency tests."""

from __future__ import annotations

import difflib
import pathlib
import platform
import sys
import time
import traceback

import numpy as np

import onnx
import onnxruntime as ort
import onnxscript

import torch

_MISMATCH_MARKDOWN_TEMPLATE = """\
### Summary

The output of ONNX Runtime does not match that of PyTorch when executing test
`{test_name}`, `sample {sample_num}` in ONNX Script `TorchLib`.

To recreate this report, use

```bash
CREATE_REPRODUCTION_REPORT=1 python -m pytest onnxscript/tests/function_libs/torch_lib/ops_test.py -k {short_test_name}
```

### ONNX Model

```
{onnx_model_text}
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

### Full error stack

```
{error_stack}
```

### Environment

```
{sys_info}
```

"""


def create_mismatch_report(
    test_name: str,
    sample_num: int,
    onnx_model: onnx.ModelProto,
    inputs,
    kwargs,
    actual,
    expected,
    error: Exception,
) -> None:
    torch.set_printoptions(threshold=sys.maxsize)

    error_text = str(error)
    error_stack = error_text + "\n" + "".join(traceback.format_tb(error.__traceback__))
    short_test_name = test_name.split(".")[-1]
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
    )

    markdown_file_name = f'mismatch-{short_test_name.replace("/", "-").replace(":", "-")}-{str(time.time()).replace(".", "_")}.md'
    markdown_file_path = save_error_report(markdown_file_name, markdown)
    print(f"Created reproduction report at {markdown_file_path}")


def save_error_report(file_name: str, text: str):
    reports_dir = pathlib.Path("error_reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    file_path = reports_dir / file_name
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(text)

    return file_path
