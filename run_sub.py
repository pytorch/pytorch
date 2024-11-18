import sys

from torch._inductor import compile_fx


idx = int(sys.argv[1])
with open(f"/tmp/aorenste/pytorch_compile_fx_tmp_input_{idx}.bin", "rb") as f:
    input = compile_fx._WireProtocolPickledInput(f.read())
result = compile_fx._SubprocessFxCompile._run_in_child(input)
with open(f"/tmp/aorenste/pytorch_compile_fx_tmp_output_{idx}.bin", "wb") as f:
    f.write(result.value)
