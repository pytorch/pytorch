import os
import contextlib
import importlib
from tabulate import tabulate
import torch
from torch._inductor.utils import fresh_inductor_cache

MODEL = "hf_Whisper"
FWD_BACKEND = "eager"
BWD_BACKEND = None
DUMP_FILENAME = f"{MODEL}-{FWD_BACKEND}_fwd-{BWD_BACKEND}_bwd.pickle"
USE_COMPILED_AUTOGRAD = BWD_BACKEND is not None
RECORD_MEMORY = False

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

        return torch.compile(gm, backend=inner_compiler, fullgraph=True, dynamic=True)

    if should_enable:
        with torch._dynamo.compiled_autograd.enable(compiler_fn) as ctx:
            yield ctx
    else:
        yield

def main(Model):
    model, example_inputs = Model('train', 'cuda').get_module()
    if FWD_BACKEND is not None:
        model = torch.compile(model, backend=FWD_BACKEND)
    x = example_inputs[0]
    with maybe_enable_compiled_autograd(USE_COMPILED_AUTOGRAD):
        out = model(x)
        loss = out.logits.sum()
        loss.backward()

if __name__ == "__main__":

    if RECORD_MEMORY:
        torch.cuda.memory._record_memory_history(
            max_entries=100000
        )

    module_name = f"torchbenchmark.models.{MODEL}"
    module = importlib.import_module(module_name)
    Model = getattr(module, "Model")

    with fresh_inductor_cache():
        main(Model)
        captures_1 = torch._dynamo.utils.counters["compiled_autograd"]["captures"]
        compiles_1 = torch._dynamo.utils.counters["compiled_autograd"]["compiles"]
        print(f"captures_1={captures_1}, compiles_1={compiles_1}")
        main(Model)
        captures_2 = torch._dynamo.utils.counters["compiled_autograd"]["captures"]
        compiles_2 = torch._dynamo.utils.counters["compiled_autograd"]["compiles"]
        print(f"captures_2={captures_2}, compiles_2={compiles_2}")

        graph_breaks = torch._dynamo.utils.counters["graph_break"]
        print(tabulate(
            [[msg, graph_breaks[msg]] for msg in graph_breaks],
            headers=["Graph Break Reason", "Count"],
        ))

    if RECORD_MEMORY:
        try:
            print(f"saving to {DUMP_FILENAME}")
            torch.cuda.memory._dump_snapshot(DUMP_FILENAME)
        except Exception as e:
            logger.error(f"Failed to capture memory snapshot {e}")
        torch.cuda.memory._record_memory_history(enabled=None)

    print(f"Memory at end of program: {torch.cuda.memory_allocated()/1e9} GB")
