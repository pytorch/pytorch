# Import generic wrappers
import transformers
from transformers import AutoModel, AutoTokenizer
from torch import _dynamo as torchdynamo

# Define the model repo
model_name = "sshleifer/tiny-gpt2"


def test_gpt2_one_shot(model_name):
    # Download pytorch model
    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Transform input tokens
    inputs = tokenizer("Hello world!", return_tensors="pt")

    # Model apply
    outputs = model(**inputs)

    # Export tiny GPT2.
    from torch.onnx._internal._fx import export, export_without_kwargs
    #onnx_model = export(model, **inputs)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    onnx_model = export_without_kwargs(model, input_ids, attention_mask)


def test_gpt2_auto_regressive(model_name):
    # NOTE: auto regressive uses generation algorithms such as greedy search or beam
    # search that involves loops and control flows.

    model = transformers.GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    input_ids = inputs["input_ids"]

    # Transform input tokens
    inputs = tokenizer("Hello world!", return_tensors="pt")

    (
        explanation,
        out_guards,
        graphs,
        ops_per_graph,
        break_reasons,
        explanation_verbose
    ) = torchdynamo.explain(model.generate, input_ids)

    print(explanation_verbose)


# test_gpt2_auto_regressive(model_name)
test_gpt2_one_shot(model_name)
