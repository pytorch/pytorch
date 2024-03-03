import os
import contextlib
import importlib
from tabulate import tabulate
import torch
from torch._inductor.utils import fresh_inductor_cache
from torch.autograd.profiler import record_function

MODEL = "t2"
FWD_BACKEND = None
BWD_BACKEND = "eager"
DUMP_FILENAME = f"{MODEL}-{FWD_BACKEND}_fwd-{BWD_BACKEND}_bwd"
USE_COMPILED_AUTOGRAD = BWD_BACKEND is not None
RECORD_MEMORY = True
PROFILE = True

"""
# figure out why this holds refs
WhisperForAudioClassification(
  (encoder): WhisperEncoder(
    (conv1): Conv1d(80, 384, kernel_size=(3,), stride=(1,), padding=(1,))
    (conv2): Conv1d(384, 384, kernel_size=(3,), stride=(2,), padding=(1,))
    (embed_positions): Embedding(1500, 384)
    (layers): ModuleList(
      (0-3): 4 x WhisperEncoderLayer(
        (self_attn): WhisperSdpaAttention(
          (k_proj): Linear(in_features=384, out_features=384, bias=False)
          (v_proj): Linear(in_features=384, out_features=384, bias=True)
          (q_proj): Linear(in_features=384, out_features=384, bias=True)
          (out_proj): Linear(in_features=384, out_features=384, bias=True)
        )
        (self_attn_layer_norm): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
        (activation_fn): GELUActivation()
        (fc1): Linear(in_features=384, out_features=1536, bias=True)
        (fc2): Linear(in_features=1536, out_features=384, bias=True)
        (final_layer_norm): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
      )                                                                                                                                                                                        )
    (layer_norm): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
  )
  (projector): Linear(in_features=384, out_features=256, bias=True)
  (classifier): Linear(in_features=256, out_features=2, bias=True)
)
"""

@contextlib.contextmanager
def maybe_enable_compiled_autograd(should_enable):
    def compiler_fn(gm):
        def inner_compiler(gm_, example_inputs_):
            torch._dynamo.utils.counters["compiled_autograd"]["compiles"] += 1
            if BWD_BACKEND == "eager":
                return torch._dynamo.backends.debugging.eager(gm_, example_inputs_)
            elif BWD_BACKEND == "aot_eager":
                return torch._dynamo.backends.debugging.aot_eager(gm_, example_inputs_)
            elif BWD_BACKEND == "inductor":
                return torch._inductor.compile(gm_, example_inputs_)
            else:
                raise ValueError(f"Unknown backend {BWD_BACKEND}")

        return torch.compile(gm, backend=inner_compiler)#, fullgraph=True, dynamic=True)

    if should_enable:
        with torch._dynamo.compiled_autograd.enable(compiler_fn) as ctx:
            yield ctx
    else:
        yield

def trace_handler(prof: torch.profiler.profile):
    # prof.export_memory_timeline(f"{DUMP_FILENAME}.html", device="cuda:0")
    prof.export_chrome_trace(f"{DUMP_FILENAME}.json.gz")

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(300, 1000),
            torch.nn.GELU(),
            torch.nn.Linear(1000, 2000),
            torch.nn.GELU(),
            torch.nn.Linear(2000, 3000),
            torch.nn.GELU(),
            torch.nn.Linear(3000, 4000),
        )

    def forward(self, x):
        return self.layers(x)

def main():
    x = torch.randn(10, 300).to("cuda")
    model = Model().to("cuda")

    if PROFILE:
        prof = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
            # profile_memory=True,
            with_stack=True,
            on_trace_ready=trace_handler,
        )
    else:
        prof = contextlib.nullcontext()

    with maybe_enable_compiled_autograd(USE_COMPILED_AUTOGRAD):
        if FWD_BACKEND is not None:
            model = torch.compile(model, backend=FWD_BACKEND)
        out = model(x)
        loss = out.sum()
        loss.backward()

        for param in model.parameters():
            param.grad.zero_()

        with prof:
            # iteration 2
            out = model(x)
            loss = out.sum()
            loss.backward()


if __name__ == "__main__":
    if RECORD_MEMORY:
        torch.cuda.memory._record_memory_history(
            max_entries=100000
        )

    with fresh_inductor_cache():
        main()

        captures = torch._dynamo.utils.counters["compiled_autograd"]["captures"]
        compiles = torch._dynamo.utils.counters["compiled_autograd"]["compiles"]
        graph_breaks = torch._dynamo.utils.counters["graph_break"]
        print(f"captures={captures}, compiles={compiles}")
        print(tabulate(
            [[msg, graph_breaks[msg]] for msg in graph_breaks],
            headers=["Graph Break Reason", "Count"],
        ))

    if RECORD_MEMORY:
        try:
            print(f"saving as {DUMP_FILENAME}.pickle")
            torch.cuda.memory._dump_snapshot(f"{DUMP_FILENAME}.pickle")
        except Exception as e:
            logger.error(f"Failed to capture memory snapshot {e}")
        torch.cuda.memory._record_memory_history(enabled=None)

    print(f"Memory at end of program: {torch.cuda.memory_allocated()/1e9} GB")
