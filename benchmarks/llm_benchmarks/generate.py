import copy
import time
from contextlib import nullcontext

from common import (
    _get_model_size,
    batch_size_combinations,
    device_sync,
    Experiment,
    get_arch_name,
    max_new_token_combinations,
    N_ITER,
)
from datasets import load_dataset
from prompts import FRANCE_ARTICLE
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

import torch


class Benchmark:
    def __init__(self, model_name, device, test_config):
        self.device = device
        self.model_name = model_name
        self.test_config = test_config

        self.model = None
        self.inputs = None
        self.dtype = None
        self.get_model_and_inputs()  # Sets self.model, self.inputs, and self.dtype
        assert self.model is not None
        assert self.inputs is not None
        assert self.dtype is not None

        self.stance = (
            torch.compiler.set_stance("force_eager")
            if test_config == "eager"
            else nullcontext()
        )

    def get_model_and_inputs(self):
        raise NotImplementedError("get_model_and_inputs() not implemented")

    def run_inference(self):
        raise NotImplementedError("run_inference() not implemented")


class ASRBenchmark(Benchmark):
    def run_inference(self):
        torch.compiler.reset()
        first_iteration = 0
        total_real_time_factor = 0
        for i in range(N_ITER):
            sample = self.inputs[i]["audio"]
            input_ids = self.processor(
                sample["array"],
                sampling_rate=sample["sampling_rate"],
                return_tensors="pt",
            )
            input_ids["input_features"] = input_ids["input_features"].to(self.device)

            device_sync(self.device)
            start = time.time()
            with self.stance:
                _ = self.model.generate(**input_ids)
            device_sync(self.device)
            end = time.time()

            input_length = len(sample["array"]) / sample["sampling_rate"]
            real_time_factor = (end - start) / input_length
            if i == 0:
                first_iteration = real_time_factor
            total_real_time_factor += real_time_factor

        avg_real_time_factor = total_real_time_factor / N_ITER

        experiment = Experiment(
            name=self.model_name,
            dtype=str(self.dtype),
            device=self.device,
            arch=get_arch_name(),
            test_config=self.test_config,
            compilation_time=first_iteration if self.test_config == "default" else 0,
            real_time_factor=avg_real_time_factor,
        )
        return experiment


class WhisperBenchmark(ASRBenchmark):
    def get_model_and_inputs(self):
        self.dtype = torch.float32
        self.inputs = load_dataset("google/fleurs", "en_us", split="validation")

        model = WhisperForConditionalGeneration.from_pretrained(self.model_name).to(
            self.device
        )
        model.forward = torch.compile(model.forward)
        model.config.forced_decoder_ids = None

        self.processor = WhisperProcessor.from_pretrained(self.model_name)
        self.model = model


class TextGenerationBenchmark(Benchmark):
    def __init__(self, model_name, device, test_config):
        super().__init__(model_name, device, test_config)

    def get_model_and_inputs(self):
        self.dtype = torch.bfloat16
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=self.dtype,
        ).to(self.device)

        model.forward = torch.compile(model.forward)
        self.model = model

        input_ids = self.tokenizer(
            FRANCE_ARTICLE * batch_size_combinations[0], return_tensors="pt"
        ).to(self.device)  # batch size =1 right now
        generation_kwargs = {
            "max_new_tokens": max_new_token_combinations[0],
            "min_new_tokens": max_new_token_combinations[0],
            "eos_token_id": None,
            "do_sample": False,
            # "cache_implementation": cache_implementation_combinations[0],
        }
        generation_config = copy.deepcopy(model.generation_config)
        generation_config.update(**generation_kwargs)

        input_ids["generation_config"] = generation_config
        self.inputs = input_ids

    def run_inference(self):
        model_size = _get_model_size(self.model)

        torch.compiler.reset()
        first_iteration = 0
        total_time = 0
        total_tokens_per_second = 0
        total_memory_bandwidth = 0
        for i in range(N_ITER):
            device_sync(self.device)
            start = time.time()
            with self.stance:
                gen_out = self.model.generate(**self.inputs)
            device_sync(self.device)
            end = time.time()
            if i == 0:
                first_iteration = end - start
            total_time += end - start
            num_tokens = len(gen_out[0]) - len(self.inputs[0])
            tokens_per_second = num_tokens / (end - start)
            total_tokens_per_second += tokens_per_second
            total_memory_bandwidth += model_size * tokens_per_second / 1e9

        avg_tokens_per_second = total_tokens_per_second / N_ITER
        avg_memory_bandwidth = total_memory_bandwidth / N_ITER

        return Experiment(
            name=self.model_name,
            dtype=str(self.dtype),
            device=self.device,
            arch=get_arch_name(),
            test_config=self.test_config,
            compilation_time=first_iteration if self.test_config == "default" else 0,
            tokens_per_second=avg_tokens_per_second,
            memory_bandwidth=avg_memory_bandwidth,
        )


def test_export_aot_inductor():
    pass


################################################################

test_configs = {
    "eager",
    "default",
    # "export-aot-inductor": test_export_aot_inductor,
}
