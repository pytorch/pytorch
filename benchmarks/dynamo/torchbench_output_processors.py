#!/usr/bin/env python3

"""Output processing functions for specific TorchBench models."""


def process_hf_reformer_output(out):
    """Process HuggingFace Reformer model output by filtering unstable elements."""
    assert isinstance(out, list)
    # second output is unstable
    return [elem for i, elem in enumerate(out) if i != 1]


def process_hf_whisper_output(out):
    """Process HuggingFace Whisper model output by filtering logits."""
    out_ret = []
    for i, elem in enumerate(out):
        if i == 0:
            if elem is not None:
                assert isinstance(elem, dict)
                out_ret.append({k: v for k, v in elem.items() if k != "logits"})
        elif i != 1:
            out_ret.append(elem)

    return out_ret


# Registry of model-specific output processing functions
PROCESS_TRAIN_MODEL_OUTPUT = {
    "hf_Reformer": process_hf_reformer_output,
    "hf_Whisper": process_hf_whisper_output,
}