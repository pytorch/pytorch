how to use lumen cli to reproduce a pytorch test failure


```bash
 uv pip install -r requirements.txt
 uv pip install -r .ci/docker/requirements-ci.txt
 uv pip download torch==2.11.0 --index-url https://download.pytorch.org/whl/test/cpu -d ./wheels

 lumen test pytorch-core \
      --group-id pytorch_jit_legacy \
      --build-env linux-jammy-py3.10-gcc11 \
```

Reproduce

```bash
export PYTHONSAFEPATH=1
 lumen test pytorch-core \
      --group-id pytorch_jit_legacy \
      --build-env linux-jammy-py3.10-gcc11 \
      --test-id jit_legacy \
      --cmd "pytest test/jit/test_tracer.py::TestTracer::test_call_traced_fn_from_traced_module -v"
```
