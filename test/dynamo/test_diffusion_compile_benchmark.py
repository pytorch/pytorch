# Owner(s): ["module: dynamo"]

import dataclasses
import json
import sys
from fnmatch import fnmatchcase
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
from torch.testing._internal.common_utils import (
    HardwareClassification,
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    skipIfTorchDynamo,
    subtest,
    TestCase,
)


REPO_ROOT = str(Path(__file__).resolve().parents[2])
sys.path.insert(0, REPO_ROOT)
try:
    from benchmarks.diffusion import compile_benchmark as benchmark
finally:
    sys.path.remove(REPO_ROOT)

# The runner compiles and counts graphs itself; wrapping tests in Dynamo makes
# its steady-state checks see the test's own compilation.
skipIfWrappedByDynamo = skipIfTorchDynamo("the benchmark runner manages compilation")


@skipIfWrappedByDynamo
class TestDiffusionCompileBenchmark(TestCase):
    hw_classification = HardwareClassification.GENERIC

    def _toy_scenarios(self, modes):
        recipe = benchmark.BENCHMARKS["toy"]
        execution = benchmark.ExecutionConfig(
            backend="eager",
            cudagraphs=False,
            warmups=0,
            repetitions=1,
            timeout_s=600,
            device="cpu",
            dtype="float32",
            num_threads=1,
        )
        return [
            benchmark.Scenario(
                recipe.model, recipe.workload, execution, mode, recipe.loader
            )
            for mode in modes
        ]

    def _success_result(self, scenario):
        return {
            "scenario_id": scenario.scenario_id,
            "status": "success",
            "error": None,
            "model": dataclasses.asdict(scenario.model),
            "workload": dataclasses.asdict(scenario.workload),
            "execution": dataclasses.asdict(scenario.execution),
            "mode": scenario.mode,
            "dtype": scenario.model.default_dtype
            if scenario.execution.dtype == "auto"
            else scenario.execution.dtype,
            "device": scenario.model.default_device
            if scenario.execution.device == "auto"
            else scenario.execution.device,
            "compiled_targets": [] if scenario.mode == "eager" else ["ToyDenoiser"],
            "model_setup_s": 0.1,
            "compile_wrapper_setup_s": 0.0,
            "compile_time_s": 0.0 if scenario.mode == "eager" else 0.5,
            "first_request_s": 0.2,
            "steady_state_samples_s": [0.3] * scenario.execution.repetitions,
            "steady_state_median_s": 0.3,
            "steady_state_mad_s": 0.0,
            "setup_device_peak": {"allocated_bytes": None, "reserved_bytes": None},
            "request_device_peak": {"allocated_bytes": None, "reserved_bytes": None},
            "compiler_diagnostics": {
                "compiler_times_s": {},
                "unique_graphs": int(scenario.mode != "eager"),
            },
            "output_summary": {
                "shape": [1],
                "dtype": "torch.float32",
                "sha256": "0" * 64,
            },
            "output_check": "not requested",
        }

    def test_request_state_is_reconstructed(self):
        class Scheduler:
            config = {"value": 1}

            def __init__(self, value):
                self.value = value

            @classmethod
            def from_config(cls, config):
                return cls(**config)

        class Pipeline:
            scheduler = Scheduler(1)

        pipeline = Pipeline()
        make_request = benchmark._fresh_request(
            pipeline, {}, 7, "cpu", mutable_parameters={"input": lambda: [1]}
        )
        _, first = make_request()
        first["input"].append(2)
        first_value = torch.rand(2, generator=first["generator"])
        first_scheduler = pipeline.scheduler
        pipeline.scheduler.value = 9
        _, second = make_request()
        second_value = torch.rand(2, generator=second["generator"])
        self.assertIsNot(pipeline.scheduler, first_scheduler)
        self.assertEqual(pipeline.scheduler.value, 1)
        self.assertEqual(second["input"], [1])
        self.assertEqual(first_value, second_value)

    # File names from the pinned revisions; "unused" files must not be downloaded.
    @parametrize(
        "name,needed,unused",
        (
            subtest(
                (
                    "auraflow",
                    ("model_index.json", "text_encoder/model.safetensors"),
                    (
                        "aura_flow_0.3.safetensors",
                        "text_encoder/model.fp16.safetensors",
                        "transformer/diffusion_pytorch_model-00001-of-00003.safetensors",
                        "vae/diffusion_pytorch_model.fp16.safetensors",
                    ),
                ),
                name="auraflow",
            ),
            subtest(
                (
                    "flux",
                    ("model_index.json", "transformer/config.json"),
                    ("flux1-dev.safetensors", "ae.safetensors"),
                ),
                name="flux",
            ),
            subtest(
                (
                    "wan",
                    ("image_encoder/model.safetensors", "vae/config.json"),
                    ("assets/logo.png", "examples/i2v_input.JPG"),
                ),
                name="wan",
            ),
            subtest(
                (
                    "ltx",
                    ("model_index.json", "vae/diffusion_pytorch_model.safetensors"),
                    ("media/trailer.gif",),
                ),
                name="ltx",
            ),
        ),
    )
    def test_prefetch_excludes_unused_weights(self, name, needed, unused):
        artifacts = benchmark.BENCHMARKS[name].model.artifacts
        snapshot = mock.Mock(return_value="snapshot")
        download = mock.Mock(return_value="file")
        hub = SimpleNamespace(snapshot_download=snapshot, hf_hub_download=download)
        with mock.patch.dict(sys.modules, {"huggingface_hub": hub}):
            benchmark._prefetch(artifacts)
        self.assertEqual(snapshot.call_args.kwargs["repo_id"], artifacts[0].repo_id)
        self.assertEqual(snapshot.call_args.kwargs["revision"], artifacts[0].revision)
        patterns = snapshot.call_args.kwargs["ignore_patterns"]
        for filename in needed:
            self.assertFalse(
                any(fnmatchcase(filename, pattern) for pattern in patterns)
            )
        for filename in unused:
            self.assertTrue(any(fnmatchcase(filename, pattern) for pattern in patterns))
        files = [
            {
                "repo_id": artifact.repo_id,
                "revision": artifact.revision,
                "filename": artifact.filename,
                "repo_type": artifact.repo_type,
            }
            for artifact in artifacts
            if artifact.filename is not None
        ]
        self.assertEqual([call.kwargs for call in download.call_args_list], files)

    def test_auraflow_uses_prefetched_config(self):
        recipe = benchmark.BENCHMARKS["auraflow"]
        scenario = dataclasses.replace(
            self._toy_scenarios(("eager",))[0],
            model=recipe.model,
            workload=recipe.workload,
            loader=recipe.loader,
        )
        serialized = json.loads(json.dumps(benchmark._scenario_to_dict(scenario)))
        self.assertEqual(benchmark._scenario_from_dict(serialized), scenario)
        base, gguf = recipe.model.artifacts
        paths = {base.key: "pinned-base", gguf.key: "pinned.gguf"}
        diffusers = SimpleNamespace(
            AuraFlowPipeline=mock.Mock(),
            AuraFlowTransformer2DModel=mock.Mock(),
            GGUFQuantizationConfig=mock.Mock(),
        )
        with (
            mock.patch.dict(sys.modules, {"diffusers": diffusers}),
            mock.patch.object(benchmark, "_finish_loading"),
        ):
            benchmark._load_auraflow(scenario, paths, "cpu", torch.float32)
        kwargs = diffusers.AuraFlowTransformer2DModel.from_single_file.call_args.kwargs
        self.assertEqual(kwargs["config"], paths[base.key])
        self.assertEqual(kwargs["subfolder"], "transformer")
        self.assertTrue(kwargs["local_files_only"])
        self.assertEqual(kwargs["torch_dtype"], torch.float32)
        diffusers.GGUFQuantizationConfig.assert_called_once_with(
            compute_dtype=torch.float32
        )
        diffusers.AuraFlowPipeline.from_pretrained.assert_called_once_with(
            paths[base.key],
            torch_dtype=torch.float32,
            transformer=diffusers.AuraFlowTransformer2DModel.from_single_file.return_value,
        )

    def test_wan_request_contract(self):
        from PIL import Image

        recipe = benchmark.BENCHMARKS["wan"]
        scenario = dataclasses.replace(
            self._toy_scenarios(("eager",))[0],
            model=recipe.model,
            workload=recipe.workload,
            loader=recipe.loader,
        )
        model, image = recipe.model.artifacts
        paths = {model.key: "pinned-model", image.key: "pinned-image.png"}
        diffusers = SimpleNamespace(
            AutoencoderKLWan=mock.Mock(), WanImageToVideoPipeline=mock.Mock()
        )
        encoder = mock.Mock()
        transformers = SimpleNamespace(CLIPVisionModel=encoder)
        pipeline = diffusers.WanImageToVideoPipeline.from_pretrained.return_value
        source = Image.new("RGB", (640, 480), "red")
        with (
            mock.patch.dict(
                sys.modules, {"diffusers": diffusers, "transformers": transformers}
            ),
            mock.patch.object(Image, "open", return_value=source) as open_image,
            mock.patch.object(benchmark, "_finish_loading") as finish,
        ):
            benchmark._load_wan(scenario, paths, "cpu", torch.bfloat16)
        open_image.assert_called_once_with(paths[image.key])
        encoder.from_pretrained.assert_called_once_with(
            paths[model.key], subfolder="image_encoder", torch_dtype=torch.float32
        )
        diffusers.AutoencoderKLWan.from_pretrained.assert_called_once_with(
            paths[model.key], subfolder="vae", torch_dtype=torch.float32
        )
        diffusers.WanImageToVideoPipeline.from_pretrained.assert_called_once_with(
            paths[model.key],
            vae=diffusers.AutoencoderKLWan.from_pretrained.return_value,
            image_encoder=encoder.from_pretrained.return_value,
            torch_dtype=torch.bfloat16,
        )
        self.assertEqual(finish.call_args.args, (pipeline, scenario, "cpu"))
        self.assertEqual(finish.call_args.kwargs["output_fields"], ("frames",))
        image_factory = finish.call_args.kwargs["mutable_parameters"]["image"]
        first = image_factory()
        first.putpixel((0, 0), (0, 0, 0))
        second = image_factory()
        self.assertEqual(second.size, (720, 544))
        self.assertEqual(second.getpixel((0, 0)), (255, 0, 0))


instantiate_parametrized_tests(TestDiffusionCompileBenchmark)


if __name__ == "__main__":
    run_tests()
