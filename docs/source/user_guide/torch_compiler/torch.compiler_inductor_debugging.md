(torch.compiler_inductor_debugging)=

# Debugging Inductor

Debugging TorchInductor can be tricky. This section covers the tools and
techniques that the team uses to gather data, which — combined with an
understanding of the underlying systems — is useful for finding and fixing
issues.

[torch.compile, the missing manual](https://docs.google.com/document/d/1y5CRfMLdwEoF1nTk9q8qEu1mgMUuUtvhklPKJ2emLU8/edit?tab=t.0#heading=h.cgice650ksgq)
has a great section on Debugging Strategies. Please read that document for an
overview of how these tools might fit into an overall debugging workflow for
`torch.compile` (and not just TorchInductor specifically).

## Compiler Bisector

The first question when root-causing issues is "Which part of the compiler is
causing this issue?" To answer, we can disable compiler subsystems one-by-one
in order to find which one is problematic. You could do this manually, but the
Compiler Bisector helps automate the process. All it requires is a function
that returns `True` upon successful compilation and execution (or whatever your
success criteria is).

### Example

Setting the config `triton.inject_relu_bug_TESTING_ONLY` will create a hidden
bug in the codegen of the RELU op. We can find this bug using the following
code:

```python
import torch
from torch._inductor.compiler_bisector import CompilerBisector

def test_fn():
    # Important: reset dynamo on every iteration
    torch._dynamo.reset()
    with config.patch("triton.inject_relu_bug_TESTING_ONLY", "accuracy"):

        def my_func(x):
            return ((x * -1) - 0.01).relu()

        inp = torch.rand([100], device="cuda")

        # Compare eager to compiled
        return torch.allclose(torch.compile(my_func)(inp), my_func(inp))

out = CompilerBisector.do_bisect(test_fn)
print(out.backend == "inductor")
print(out.subsystem == "lowerings")
print(out.bisect_number == 2)
print("relu" in out.debug_info)
```

### Subsystems

Below is a summary of the subsystems that the Bisector will disable during its
exploration. These subsystems are specified in the `BACKENDS` variable in
`compiler_bisector.py`.

| Subsystem | What It Controls |
|-----------|-----------------|
| `eager` | Run Dynamo without aot_autograd. |
| `aot_eager` | Run Dynamo with aot_autograd, but without Min-Cut Partitioner or Decomps. |
| `aot_eager_decomp_partition` | Run Dynamo with aot_autograd, decompositions and partitioner. |
| `aot_eager_decomp_partition.cse` | [Common Subexpression Elimination](https://en.wikipedia.org/wiki/Common_subexpression_elimination) pass. |
| `aot_eager_decomp_partition.decomposition` | The number of decompositions we apply in tracing. |
| `aot_eager_decomp_partition_crossref` | Applies CrossRefFakeMode to check that the graph is well formed. |
| `inductor.joint_graph_passes` | AOTAutograd Graph Passes applied to the Joint Graph. |
| `inductor.post_grad_passes` | AOTAutograd Graph Passes applied individually on forward, and backward graph. |
| `inductor.fallback_random` | Fallback to eager for random/dropout. |
| `inductor.emulate_precision_casts` | Mode to emulate PyTorch eager numerics when doing lower precision compute. |
| `inductor.layout_optimization` | Whether or not to do layout optimization on Convolution operations. |
| `inductor.comprehensive_padding` | Whether or not we do padding on flexible layouts. |
| `inductor.lowerings` | Whether or not we lower ops or use ATen fallback. |

## tlparse/TORCH_TRACE

Enabling `TORCH_TRACE` causes the compiler to dump structured logs for various
operations. `tlparse` is a tool that parses those logs and generates html that
categorizes and makes it easy to view those logs. Note that
[torch.compile, the missing manual](https://docs.google.com/document/d/1y5CRfMLdwEoF1nTk9q8qEu1mgMUuUtvhklPKJ2emLU8/edit?tab=t.0#heading=h.u9711c2s6w0c)
contains a section on tlparse / TORCH_TRACE. Here, we focus on covering the
contents of TORCH_TRACE that are most relevant to TorchInductor.

To get started, install tlparse:

```bash
$ pip install tlparse
```

We'll be using this simple test script:

```python
import torch

def myfn(in1, in2):
    tmp = in1.sin() @ in2
    return tmp.cos()

compiled = torch.compile(myfn)

t1 = torch.randn(1024, 256)
t2 = torch.randn(256, 1024)
result = compiled(t1, t2)
print(result.shape)
```

We can run the script with `TORCH_TRACE` pointing to an output directory, then
use `tlparse` to process the logs. We also disable caches so that we ensure we
get the full trace of the compilation:

```bash
$ TORCHINDUCTOR_FORCE_DISABLE_CACHES=1 TORCH_TRACE="./logs" python script.py
torch.Size([1024, 1024])
$ ls ./logs
dedicated_log_torch_trace_4dke989s.log
$ tlparse logs/dedicated_log_torch_trace_4dke989s.log -o output

  [00:00:00] [######################################################################] 146.38 KiB/146.38 KiB [14.01 MiB/s] (0s)

  Stats { ok: 34, other_rank: 0, fail_glog: 0, fail_json: 0, fail_payload_md5: 0, fail_dynamo_guards_json: 0, fail_parser: 0, unknown: 0 }
  Stats { ok: 35, other_rank: 0, fail_glog: 0, fail_json: 0, fail_payload_md5: 0, fail_dynamo_guards_json: 0, fail_parser: 0, unknown: 0 }
$ ls output
-_0_0_0  chromium_events.json  compile_directory.json  failures_and_restarts.html  index.html  raw.log
```

`index.html` renders like this:

```{image} ../../_static/img/inductor_user_guide/tlparse_index.png
:alt: tlparse index.html showing Stack Frames and compile artifacts organized by compile ID
:width: 600px
:align: center
```

This view shows different **Stack Frames**, which
together form a **Stack Trie**. Each of these frames correspond to one or more
[compile](https://github.com/pytorch/pytorch/blob/3d06ff82a84a118f0ed246864d4fc01ac4726328/torch/_inductor/__init__.py#L33)
calls in TorchInductor, and you can see the intermediate results from
TorchInductor below. Looking at the value `[0/0]`, the first number is the
index of the stack frame which was compiled, and the second number is the nth
time that this frame was compiled. These recompilations usually happen because
the compiler's previous assumptions about your code were violated by a new
input shape, and a
[Guard](https://docs.pytorch.org/docs/stable/torch.compiler_dynamic_shapes.html#the-guard-model)
was triggered.

You can view timing information by going to <https://ui.perfetto.dev/> and
opening up the file `chromium_events.json`:

```{image} ../../_static/img/inductor_user_guide/perfetto_timing.png
:alt: Perfetto timing view showing runtime breakdown of compiler phases
:width: 600px
:align: center
```

This view breaks down the runtime
of various important phases in the compiler and can be helpful when debugging
performance issues in the compilation itself.

**`0_0_0/dynamo_output_graph_0.txt`** contains the
[GraphModule](https://docs.pytorch.org/docs/stable/fx.html#torch.fx.GraphModule)
that Dynamo passes to TorchInductor. It's good to check this first to see if
the Graph has been altered by Dynamo in an unexpected way:

```python
class GraphModule(torch.nn.Module):
    def forward(self, L_in1_: "f32[1024, 256][256, 1]cpu", L_in2_: "f32[256, 1024][1024, 1]cpu"):
        l_in1_ = L_in1_
        l_in2_ = L_in2_

        # File: /home/user/org/doc-tutorial/tlparse.py:4 in myfn, code: tmp2 = in1.sin() @ in2
        sin: "f32[1024, 256][256, 1]cpu" = l_in1_.sin();  l_in1_ = None
        tmp2: "f32[1024, 1024][1024, 1]cpu" = sin @ l_in2_;  sin = l_in2_ = None

        # File: /home/user/org/doc-tutorial/tlparse.py:5 in myfn, code: return tmp2.cos()
        cos: "f32[1024, 1024][1024, 1]cpu" = tmp2.cos();  tmp2 = None
        return (cos,)
```

Looks like what we expect!

Moving on to AOTAutograd, the file
**`_0_0_0/aot_forward_graph_fw_metadata_4.txt`** contains all of the
aliasing/mutation metadata that AOTAutograd tracks. The description of what
these fields mean is
[here](https://github.com/pytorch/pytorch/blob/5c79a55e7e58c6382c7ce02da1cd07d358239d94/torch/_functorch/_aot_autograd/schemas.py#L362).

### Debugging Inductor-Generated Code

TorchInductor accepts an FX graph and produces optimized code, so most of the
issues you may encounter in TorchInductor are caused by generated code that
doesn't match your expectations. Therefore, one of the most important debugging
techniques for TorchInductor is to take the codegen'd output code, and alter it
as you see fit. The output code is found in a file that looks like:
**`_0_0_0/inductor_output_code_cku5wtbugds5uhixzrrzlzwg2auzunvwpdzyyfcecnwwcctgh4js_6.html`**

Here, you can see not only the kernels being run by Inductor, but also how they
are being called in sequence. Suppose you want to see the output of the sin
operation, for example. Let's add a breakpoint to `call` function:

```python
def call(args):
    arg0_1, arg1_1 = args
    args.clear()
    assert_size_stride(arg0_1, (1024, 256), (256, 1))
    assert_size_stride(arg1_1, (256, 1024), (1024, 1))
    buf0 = empty_strided_cpu((1024, 256), (256, 1), torch.float32)
    cpp_fused_sin_0(arg0_1, buf0)
    del arg0_1

    breakpoint() # <-- HERE

    buf1 = empty_strided_cpu((1024, 1024), (1024, 1), torch.float32)
    # Topologically Sorted Source Nodes: [sin, tmp2], Original ATen: [aten.sin, aten.mm]
    extern_kernels.mm(buf0, arg1_1, out=buf1)
    del arg1_1
    del buf0
    buf2 = buf1; del buf1  # reuse
    cpp_fused_cos_1(buf2)
    return (buf2, )
```

```bash
$ python output_code.py

> /home/user/org/doc-tutorial/output_code.py(96)call()
-> buf1 = empty_strided_cpu((1024, 1024), (1024, 1), torch.float32)
(Pdb) buf0
tensor([[ 0.3684, -0.0135, -0.9576,  ...,  0.1886, -0.3092, -0.9964],
        [-0.1938, -0.6284, -0.6756,  ..., -0.9303,  0.6191, -0.8744],
        [ 0.1716,  0.3536,  0.1611,  ...,  0.2880,  0.8847,  0.8116],
        ...,
        [-0.0853,  0.7208, -0.2777,  ..., -0.9894,  0.4127,  0.6757],
        [-0.5918,  0.5848,  0.2671,  ..., -0.6380,  0.9973,  0.4200],
        [-0.2439, -0.9298, -0.8265,  ...,  0.7463, -0.7712,  0.2884]])
```

Or, suppose we want to change the generated kernel from sin to arcsin. We can
just edit the kernel string directly:

```cpp
cpp_fused_sin_0 = async_compile.cpp_pybinding(['const float*', 'float*'], '''
#include "/tmp/torchinductor_user/pi/cpicxudqmdsjh5cm4klbtbrvy2cxwr7whxl3md2zzdjdf3orvfdf.h"
extern "C"  void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(46)
    {
        int tid = omp_get_thread_num();
        {
            #pragma omp for
            for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(262144L); x0+=static_cast<int64_t>(16L))
            {
                {
                    if(C10_LIKELY(x0 >= static_cast<int64_t>(0) && x0 < static_cast<int64_t>(262144L)))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<int64_t>(x0), static_cast<int64_t>(16));
                        // HERE
                        auto tmp1 = tmp0.asin(); // sin -> asin
                        // HERE
                        tmp1.store(out_ptr0 + static_cast<int64_t>(x0));
                    }
                }
            }
        }
    }
}
''')
```

Once you've figured out how you want to change the output code in your specific
instance, you can work backwards from there to find the part of TorchInductor
that needs to change to produce that output code generally.

## TORCH_LOGS

There is some feature overlap with
[TORCH_LOGS](https://docs.pytorch.org/tutorials/recipes/torch_logs.html) and
`TORCH_TRACE`, but the former still plays an important role for gathering
debugging information quickly. To enable logging for a specific module, set a
value like `TORCH_LOGS=+torch._inductor.codecache`.

`TORCH_LOGS=+all` will print everything. Setting `TORCH_LOGS=help` describes
all available logging options.

## Debugging Performance Issues

[torch.compile: the missing manual](https://docs.google.com/document/d/1y5CRfMLdwEoF1nTk9q8qEu1mgMUuUtvhklPKJ2emLU8/edit?tab=t.0#heading=h.cfjs9aarcek)
has a very thorough section on this, and TorchInductor-specific techniques are
covered there.

### GEMM Performance

Matrix Multiplies make up a large percentage of cycles through machine learning
systems, so there are many options for improving GEMM Performance. Here is a
list of things to try:

- [Turn on Max-Autotune](https://github.com/pytorch/pytorch/blob/86eb65f7f06016bcd5d7951dc9d74bc3993a827a/torch/_inductor/config.py#L429)
- [Turn on Coordinate Descent tuning](https://github.com/pytorch/pytorch/blob/86eb65f7f06016bcd5d7951dc9d74bc3993a827a/torch/_inductor/config.py#L521)
- Turn on both Max-Autotune and Coordinate Descent tuning
- [Turn on Exhaustive Autotuning (expensive)](https://github.com/pytorch/pytorch/blob/50f23ff6f883db5021dd6bab4c146434f98dd15d/torch/_inductor/config.py#L483)
- If you think there's a better config that we're not trying in the exhaustive
  list,
  [add it to this list with Max-Autotune turned on](https://github.com/pytorch/pytorch/blob/86eb65f7f06016bcd5d7951dc9d74bc3993a827a/torch/_inductor/template_heuristics.py#L164)
- [Try the TMA template](https://github.com/pytorch/pytorch/blob/50f23ff6f883db5021dd6bab4c146434f98dd15d/torch/_inductor/config.py#L1348)
- [Try CUTLASS](https://github.com/pytorch/pytorch/blob/50f23ff6f883db5021dd6bab4c146434f98dd15d/torch/_inductor/config.py#L467)

## Debugging Numerical Issues

Inductor does not guarantee numerical equivalence with Eager mode, mainly
because operations can be reordered and
[Floating Point Arithmetic is not Associative](https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html).
In order to establish that a numerical issue does indeed exist, the standard
way to compare tensors between eager and TorchInductor is to run your model in
fp64, and then compare the eager output and the TorchInductor output to the
fp64 baseline.

### Emulate Precision Casts

The Eager behavior of computing bf16/fp16 is to upcast inputs to fp32, and then
downcasting after, which TorchInductor does not do. The debugging flag
[inductor.emulate_precision_casts](https://github.com/pytorch/pytorch/blob/9620994067b18e846a097d1e99af85ec2426ef0a/torch/_inductor/config.py#L680)
emulates this behavior. ***Note that this setting can lower performance and
generally has worse numerics than the default behavior.***

## Debugging NaN Issues

The first step in fixing a NaN issue is to find the first tensor in the model
that contains a NaN, as NaNs will multiply. The best tool for this is
`TORCHINDUCTOR_NAN_ASSERTS`. This will codegen asserts into the kernel that
check for NaNs as the kernel is executing.

### TORCHINDUCTOR_NAN_ASSERTS

Let's create an example model that will cause NaNs:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class NaNModel(nn.Module):
    """
    Neural network that creates NaNs through log of negative numbers.
    """

    def __init__(self, input_size=10, hidden_size=64):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, hidden_size)
        self.linear4 = nn.Linear(hidden_size, hidden_size)
        self.linear5 = nn.Linear(hidden_size, hidden_size)
        self.linear6 = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = F.relu(self.linear4(x))
        x = F.relu(self.linear5(x))

        # Create NaN through log of negative numbers
        x = torch.log(x - 10)
        x = F.relu(self.linear6(x))

        x = self.output(x)
        x = torch.sum(x)

        return x


if __name__ == "__main__":
    model = NaNModel()
    x = torch.randn(3, 10) + 5  # Positive values to trigger log(negative)

    compiled_model = torch.compile(model)

    output = compiled_model(x)
    print(f"Output: {output}")
```

Now, let's run the script with `TORCHINDUCTOR_NAN_ASSERTS`:

```bash
$ TORCHINDUCTOR_NAN_ASSERTS=1 python nan_model.py
...
  File "/tmp/torchinductor_user/b4/cb4bljzelkiex7nqbudphcxpij6bpkpy6hjcpxtqayj47wzsj3ns.py", line 291, in call
    assert not buf9.isnan().any().item()
AssertionError
```

Opening this file takes us to the Inductor generated code, where we can see the
offending operation:

```python
            stream0 = get_raw_stream(0)
            triton_poi_fused_log_relu_sub_1.run(buf8, buf9, 192, stream=stream0)
            assert not buf8.isnan().any().item()
            assert not buf8.isinf().any().item()
            assert not buf9.isnan().any().item() # Here
            assert not buf9.isinf().any().item()
```

Looking at the comment above, the kernel tells us which ops created the NaN:

```python
# kernel path: /tmp/torchinductor_user/sy/csywiifufcvhhvhxgw25enjcj3r77mw3yimw7luwtub5zeqgfnx5.py
# Topologically Sorted Source Nodes: [x_4, sub, x_5], Original ATen: [aten.relu, aten.sub, aten.log]
# Source node to ATen node mapping:
#   sub => sub
#   x_4 => relu_4
#   x_5 => log
# Graph fragment:
#   %addmm_4 : Tensor "f32[3, 64][64, 1]cuda:0" = PlaceHolder[target=addmm_4]
#   %relu_4 : Tensor "f32[3, 64][64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%addmm_4,), kwargs = {})
#   %sub : Tensor "f32[3, 64][64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%relu_4, 10), kwargs = {})
#   %log : Tensor "f32[3, 64][64, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.log.default](args = (%sub,), kwargs = {})
#   return %log
triton_poi_fused_log_relu_sub_1 = async_compile.triton('triton_poi_fused_log_relu_sub_1', '''
```

You can even set a breakpoint in the file, and either rerun the original script,
or run the file directly:

```bash
$ TORCHINDUCTOR_NAN_ASSERTS=1 python /tmp/torchinductor_user/b4/cb4bljzelkiex7nqbudphcxpij6bpkpy6hjcpxtqayj47wzsj3ns.py
> /tmp/torchinductor_user/b4/cb4bljzelkiex7nqbudphcxpij6bpkpy6hjcpxtqayj47wzsj3ns.py(292)call()
-> assert not buf9.isnan().any().item()
(Pdb) buf8
tensor([[ -381.1685,  2756.5942,  2190.4092,  1450.5792,  2567.5635,   632.5462,
          -460.6224,   550.8149,  2141.5667,  1800.2137,  2502.6318,  -522.3486,
          3353.0259,  1392.3037,   538.2896,  1603.8689,  -303.1974, -1590.7761,
         -2318.3445,   -59.6752, -1585.8706,   476.3831,  -756.5652,   175.9389,
         -2664.4622, -2057.7871,  3624.0811,  1738.3439,   942.6910,  2516.7864,
           460.2427,  1120.9181, -3562.1121, -1712.9216,  -262.1320, -1418.2817,
         -2941.0078, -1802.9307, -1855.0543,  -166.1157,    75.4381, -2558.2244,
           826.9001,  1319.3993,  1422.1490, -1388.0497,   691.9113,   747.3169,
         -2790.7976,  1679.3070,  -889.8872,  -416.5905,  -851.2183,  -866.2843,
         -1832.8572,   593.8383,  1746.9064, -3428.0256, -1926.7656, -1177.8613,
           309.1942,  2341.3616, -3086.9873,   243.2090],
        [-2729.1016, -1294.1569,  1493.4093,  2523.1641, -1274.7378,  1068.9348,
         -1980.6021, -1190.8618,  1998.7396,  1662.1578,  -458.9829,  2990.4011,
          3611.5686,  1602.6514,   125.1075,  2486.0146, -1746.8807,   679.7756,
          -133.9449,   958.6292, -2968.4187,  2991.4883, -3062.1091,  4920.3862,
         -1886.1661,   -51.4195,  1452.5326,  3251.5422, -3056.2085,  2783.9045,
          1316.2456,  2676.7642, -1742.9053,  2905.4683,  2984.5510,   -69.6492,
         -3584.2905,  -144.8501, -2676.7351, -1569.7478,  3616.3809, -1232.4596,
           433.6350,  -128.2836,  5850.2412,  1255.1045,   290.5077, -3442.6719,
          -864.2769,   629.6766,  4592.3667, -1998.6271,  -600.8240, -1575.7009,
          -861.3682, -3182.1311,  6895.9854,  -880.6266,  -850.1956,   683.5182,
         -2681.4331,   313.4224, -1574.2748,  1754.2289],
        [-5973.4722,  1172.0880,  -409.8680,   754.3694,  -937.7782,   -69.4745,
         -4151.4019, -2218.8467,  6324.3901,   777.9156,  4511.8403,   151.3943,
          2635.0757,  3903.4363,  -529.5245,  1139.8651,   -82.3232,  2118.1675,
            10.0820,  1448.0802, -8195.2754,  1562.3729, -2005.2090,  4502.0698,
         -1486.9825, -3523.6855,  2798.3374,  1246.1573, -4956.0293,  3095.7600,
          -591.0005,   991.7573, -5951.9976,  1268.5929,   628.1058, -1611.2432,
         -2916.5254,  1506.2789, -2105.1660, -1438.6881,  2806.6704, -8616.0098,
          2705.4253,   -38.6298,  2047.4551,  1165.5925,  -687.3981, -5377.9058,
          -237.3144,   642.8304,   911.9037, -3429.9312, -2860.1619, -4408.0986,
         -3557.2361, -3360.8367,  3463.3147, -3685.0305, -4805.5645, -2073.8533,
         -2262.8882,  5443.8223,  1134.6807, -2385.6345]], device='cuda:0')
(Pdb) buf9
tensor([[    nan,  7.9181,  7.6873,  7.2728,  7.8468,  6.4338,     nan,  6.2931,
          7.6646,  7.4901,  7.8211,     nan,  8.1146,  7.2315,  6.2696,  7.3739,
             nan,     nan,     nan,     nan,     nan,  6.1450,     nan,  5.1116,
             nan,     nan,  8.1926,  7.4549,  6.8381,  7.8268,  6.1098,  7.0129,
             nan,     nan,     nan,     nan,     nan,     nan,     nan,     nan,
          4.1811,     nan,  6.7055,  7.1773,  7.2529,     nan,  6.5249,  6.6030,
             nan,  7.4202,     nan,     nan,     nan,     nan,     nan,  6.3696,
          7.4599,     nan,     nan,     nan,  5.7011,  7.7542,     nan,  5.4519],
        [    nan,     nan,  7.3021,  7.8293,     nan,  6.9650,     nan,     nan,
          7.5953,  7.4098,     nan,  7.9998,  8.1891,  7.3732,  4.7459,  7.8144,
             nan,  6.5069,     nan,  6.8550,     nan,  8.0002,     nan,  8.4991,
             nan,     nan,  7.2742,  8.0838,     nan,  7.9280,  7.1749,  7.8886,
             nan,  7.9709,  7.9978,     nan,     nan,     nan,     nan,     nan,
          8.1905,     nan,  6.0489,     nan,  8.6725,  7.1270,  5.6366,     nan,
             nan,  6.4292,  8.4300,     nan,     nan,     nan,     nan,     nan,
          8.8372,     nan,     nan,  6.5125,     nan,  5.7151,     nan,  7.4641],
        [    nan,  7.0580,     nan,  6.6125,     nan,     nan,     nan,     nan,
          8.7506,  6.6437,  8.4122,  4.9516,  7.8729,  8.2670,     nan,  7.0299,
             nan,  7.6536, -2.5016,  7.2711,     nan,  7.3475,     nan,  8.4101,
             nan,     nan,  7.9332,  7.1198,     nan,  8.0346,     nan,  6.8893,
             nan,  7.1377,  6.4267,     nan,     nan,  7.3107,     nan,     nan,
          7.9362,     nan,  7.8993,     nan,  7.6195,  7.0524,     nan,     nan,
             nan,  6.4502,  6.8045,     nan,     nan,     nan,     nan,     nan,
          8.1471,     nan,     nan,     nan,     nan,  8.6004,  7.0253,     nan]],
       device='cuda:0')
```

Looks like all the negative numbers got converted to NaNs!

## Debugging Config Issues

[Inductor has a lot of configs](https://github.com/pytorch/pytorch/blob/9620994067b18e846a097d1e99af85ec2426ef0a/torch/_inductor/config.py).
We have strong CI coverage, but the reality is that the config space is so
large that no CI could cover the powerset of possible options. If you have a lot
of non-default configs, and you suspect that one of them is causing a bug,
consider using the `ConfigFuzzer` to find the faulty config:

```python
import torch._inductor.config as cfg


def create_simple_test_model_gpu():
    batch_size = 32
    seq_length = 50
    hidden_size = 768

    def test_fn() -> bool:
        inp = torch.randn(batch_size, seq_length, hidden_size, device="cuda")
        weight = torch.randn(hidden_size, hidden_size, device="cuda")
        matmul_output = inp @ weight
        final_output = torch.nn.LayerNorm(hidden_size, device="cuda")(matmul_output)
        return True

    return test_fn


fuzzer = ConfigFuzzer(cfg, create_simple_test_model_gpu, seed=2)

# Test random configs with bisection:
failing_configs = fuzzer.bisect(num_attempts=10)

# reproduce a failing config
fuzzer.reproduce(
    [{"triton.autotune_pointwise": ..., "coordinate_descent_tuning": ...}]
)
```

`failing_configs` will contain a list of minified configs that cause
`create_simple_test_model_gpu` to fail. **Make sure to first run this on
main to make sure that the failures are unique to your branch!**

## Accuracy Debugging using Fuzzers

When evaluating accuracy issues between eager execution and `torch.compile` in
PyTorch, always establish a high-precision ground truth baseline (FP64 or FP32),
then create both BF16 eager and compile variants from the same base model to
ensure fair comparison. Use `torch._dynamo.utils.same()` as the primary
evaluation method (maintaining default tolerances (1e-4) without modification),
and avoid comparing indices from max/min operations as they bypass tolerance
requirements. Supplement the analysis with MSE comparisons between eager and
compile results, and both variants against the FP64 baseline to quantify
numerical differences.

**Example:**

```python
# 1. Load models with different precisions
model_baseline = load_model(torch_dtype=torch.float64)  # FP64 ground truth
model_bf16 = load_model(torch_dtype=torch.bfloat16)     # BF16 base model

# 2. Create eager and compile variants from same base model
model_bf16_eager = copy.deepcopy(model_bf16)
model_bf16_compile = copy.deepcopy(model_bf16)
model_bf16_compile.forward = torch.compile(model_bf16_compile.forward)

# 3. Run inference and compare using fuzzer protocol
same_result = same(eager_results, compile_results, fp64_baseline, tol=1e-4)

# 4. Supplementary MSE analysis for deeper debugging
mse_eager_vs_compile = nn.MSELoss()(eager_f32, compile_f32)
mse_eager_vs_baseline = nn.MSELoss()(eager_f32, baseline_f32)
mse_compile_vs_baseline = nn.MSELoss()(compile_f32, baseline_f32)
```

**Example Code for Llama3.2-1B model:**

```python
import copy
import time
import unittest

import torch
from torch import nn
from torch._dynamo.utils import same
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "meta-llama/Llama-3.2-1B"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# FP64 baseline model (highest precision, used as ground truth)
model_baseline = AutoModelForCausalLM.from_pretrained(
   model_name, torch_dtype=torch.float64, device_map="cuda"
)

# BF16 base model (will be copied for eager and compile variants)
model_bf16 = AutoModelForCausalLM.from_pretrained(
   model_name, torch_dtype=torch.bfloat16, device_map="cuda"
)

# Set eval mode
model_baseline.eval()
model_bf16.eval()

# Configure generation settings for all models
generation_config = {
   "do_sample": False,
   "use_cache": True,
   "cache_implementation": "static",
   "max_new_tokens": 2000,
   "pad_token_id": tokenizer.eos_token_id,
   "temperature": 0.0,
}

for key, value in generation_config.items():
   setattr(model_baseline.generation_config, key, value)
   setattr(model_bf16.generation_config, key, value)

vocab_size = tokenizer.vocab_size
input_ids = torch.randint(
   low=0,
   high=vocab_size,
   size=(1, 1000),
   device="cuda",
   dtype=torch.long,
)
example_inputs = {"input_ids": input_ids}

# Get FP64 baseline results (ground truth) - no copy needed
start = time.time()
baseline_results = model_baseline(**example_inputs).logits
print(f"FP64 baseline time: {time.time() - start}")

# Get BF16 eager results
model_bf16_eager = copy.deepcopy(model_bf16)
start = time.time()
bf16_eager_results = model_bf16_eager(**example_inputs).logits
print(f"BF16 eager time: {time.time() - start}")

# Get BF16 compile results
model_bf16_compile = copy.deepcopy(model_bf16)
model_bf16_compile.forward = torch.compile(model_bf16_compile.forward)
start = time.time()
bf16_compile_results = model_bf16_compile(**example_inputs).logits
print(f"BF16 compile time: {time.time() - start}")

print("\n=== Fuzzer Protocol: torch._dynamo.utils.same (with fp64_ref) ===")

# Fuzzer protocol: use default tolerance and fp64_ref
same_eager_vs_compile = same(
   bf16_eager_results,  # ref (bf16 eager)
   bf16_compile_results,  # res (bf16 compile)
   baseline_results,  # fp64_ref (high precision ground truth)
   tol=1e-4,  # default tolerance - do not modify
   exact_dtype=False,  # allow different dtypes
)

print(f"torch._dynamo.utils.same result: {'PASS' if same_eager_vs_compile else 'FAIL'}")

print("\n=== MSE Comparison ===")

# MSE comparison (convert to same dtype for fair comparison)
loss = nn.MSELoss()
bf16_eager_results_f32 = bf16_eager_results.to(dtype=torch.float32)
bf16_compile_results_f32 = bf16_compile_results.to(dtype=torch.float32)

mse_eager_vs_compile = loss(bf16_compile_results_f32, bf16_eager_results_f32)
print(f"MSE between BF16 eager and BF16 compile: {mse_eager_vs_compile}")

# Compare each model against fp64 baseline
baseline_results_f32 = baseline_results.to(dtype=torch.float32)
mse_eager_vs_baseline = loss(bf16_eager_results_f32, baseline_results_f32)
mse_compile_vs_baseline = loss(bf16_compile_results_f32, baseline_results_f32)

print(f"MSE BF16 eager vs FP64 baseline: {mse_eager_vs_baseline}")
print(f"MSE BF16 compile vs FP64 baseline: {mse_compile_vs_baseline}")
```
