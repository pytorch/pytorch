# Owner(s): ["module: dynamo"]

import dataclasses
import json
import sys
import weakref
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

    def test_invalid_mode_and_missing_region(self):
        model = torch.nn.Linear(2, 2)
        with self.assertRaisesRegex(ValueError, "invalid compilation mode"):
            with benchmark.configure_compilation(
                model, "invalid", ("Linear",), "eager", False
            ):
                pass
        with self.assertRaisesRegex(ValueError, "no repeated compilation targets"):
            with benchmark.configure_compilation(
                model, "regional", ("MissingBlock",), "eager", False
            ):
                pass
        with self.assertRaisesRegex(ValueError, "CUDA graphs require the inductor"):
            benchmark._compile_kwargs("eager", True)

        scenario = self._toy_scenarios(("eager",))[0]
        invalid_workload = dataclasses.replace(
            scenario.workload, output_boundary="unspecified"
        )
        with self.assertRaisesRegex(ValueError, "invalid output boundary"):
            benchmark._validate_worker_scenario(
                dataclasses.replace(scenario, workload=invalid_workload), "cpu"
            )

    def test_regional_prefers_model_entry_point(self):
        class Block(torch.nn.Module):
            def forward(self, value):
                return value + 1

        class Model(torch.nn.Module):
            _repeated_blocks = ("Block",)

            def __init__(self):
                super().__init__()
                self.block = Block()
                self.called = False

            def compile_repeated_blocks(self, **kwargs):
                self.called = True

        model = Model()
        with benchmark.configure_compilation(
            model, "regional", ("Block",), "eager", False
        ) as details:
            self.assertTrue(model.called)
            self.assertEqual(
                details["regional_entry_point"], "model.compile_repeated_blocks"
            )

    def test_regional_falls_back_to_explicit_classes(self):
        class Block(torch.nn.Module):
            def forward(self, value):
                return value + 1

        class Mismatched(torch.nn.Sequential):
            _repeated_blocks = ("Other",)

            def compile_repeated_blocks(self, **kwargs):
                raise AssertionError("mismatched entry point must not be used")

        model = torch.nn.Sequential(Block(), Block())
        with mock.patch.object(torch.nn.Module, "compile") as compile_module:
            with benchmark.configure_compilation(
                model, "regional", ("Block",), "eager", False
            ) as details:
                pass
        self.assertEqual(details["regional_entry_point"], "explicit class selection")
        self.assertEqual(compile_module.call_count, 2)
        with self.assertRaisesRegex(ValueError, "do not match recipe"):
            with benchmark.configure_compilation(
                Mismatched(Block()), "regional", ("Block",), "eager", False
            ):
                pass

    def test_hierarchical_marking_is_idempotent_and_restored(self):
        class Block(torch.nn.Module):
            def forward(self, value):
                return value + 1

        model = torch.nn.Sequential(Block(), Block())
        original = Block.forward
        with benchmark.configure_compilation(
            model, "hierarchical", ("Block",), "eager", False
        ):
            marked = Block.forward
            self.assertTrue(hasattr(marked, "__marked_compile_region_fn__"))
            with benchmark.configure_compilation(
                model, "hierarchical", ("Block",), "eager", False
            ):
                self.assertIs(Block.forward, marked)
            self.assertIs(Block.forward, marked)
        self.assertIs(Block.forward, original)

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

    def test_device_initializes_before_setup_timing(self):
        scenario = self._toy_scenarios(("eager",))[0]
        scenario = dataclasses.replace(
            scenario,
            execution=dataclasses.replace(
                scenario.execution, device="cuda:0", num_threads=None
            ),
        )
        calls = []
        resolved = ("cuda:0", "float32", torch.float32)

        def record(name):
            return lambda device: calls.append((name, device))

        with (
            mock.patch.object(
                benchmark, "_resolve_device_and_dtype", return_value=resolved
            ),
            mock.patch.object(
                torch, "empty", side_effect=lambda size, device: record("init")(device)
            ),
            mock.patch.object(benchmark, "_synchronize", side_effect=record("sync")),
            mock.patch.object(
                benchmark, "_reset_device_memory", side_effect=record("reset")
            ),
            mock.patch.object(
                benchmark, "_load_toy", side_effect=RuntimeError("reached model setup")
            ),
            self.assertRaisesRegex(RuntimeError, "reached model setup"),
        ):
            benchmark.execute_scenario(scenario)
        # Synchronizing and resetting peaks do not start the runtime of an indexed
        # device, so without the explicit initialization setup would include it.
        expected = [("init", "cuda:0"), ("sync", "cuda:0"), ("reset", "cuda:0")]
        self.assertEqual(calls, expected)

    @parametrize("cudagraphs", (True, False))
    def test_timed_request_cudagraph_boundary(self, cudagraphs):
        output = object()

        def make_request():
            mark.assert_not_called()
            return (), {}

        def pipeline():
            self.assertEqual(mark.call_count, int(cudagraphs))
            return output

        loaded = benchmark.LoadedPipeline(pipeline, None, make_request, ())
        with mock.patch.object(torch.compiler, "cudagraph_mark_step_begin") as mark:
            for _ in range(3):
                mark.reset_mock()
                _, result = benchmark._timed_request(
                    loaded, "cpu", cudagraphs=cudagraphs
                )
                self.assertIs(result, output)
                self.assertEqual(mark.call_count, int(cudagraphs))

    def test_measured_outputs_are_released_between_requests(self):
        scenario = self._toy_scenarios(("eager",))[0]
        scenario = dataclasses.replace(
            scenario,
            execution=dataclasses.replace(
                scenario.execution, warmups=1, repetitions=3, num_threads=None
            ),
        )
        outputs = []

        def pipeline():
            if outputs:
                self.assertIsNone(outputs[-1]())
            value = torch.ones(1)
            outputs.append(weakref.ref(value))
            return {"images": value}

        loaded = benchmark.LoadedPipeline(
            pipeline, torch.nn.Identity(), lambda: ((), {}), ("images",)
        )
        with (
            mock.patch.object(benchmark, "_load_toy", return_value=loaded),
            torch.random.fork_rng(devices=[]),
        ):
            benchmark.execute_scenario(scenario)
        self.assertEqual(len(outputs), 5)

    def test_steady_state_recompilation_is_rejected(self):
        scenario = self._toy_scenarios(("full",))[0]
        scenario = dataclasses.replace(
            scenario,
            execution=dataclasses.replace(
                scenario.execution, repetitions=2, num_threads=None
            ),
        )
        model = torch.nn.Linear(4, 4)
        rows = iter((1, 2, 3))

        def make_request():
            return (torch.randn(next(rows), 4),), {}

        loaded = benchmark.LoadedPipeline(model, model, make_request, ())
        with (
            mock.patch.object(benchmark, "_load_toy", return_value=loaded),
            self.assertRaisesRegex(RuntimeError, "during steady-state measurement"),
        ):
            benchmark.execute_scenario(scenario)

    @parametrize("case", ("captured", "empty", "skipped"))
    def test_cudagraph_diagnostics(self, case):
        from torch._dynamo.utils import counters

        captured = case != "empty"
        leaf = SimpleNamespace(graph=object() if captured else None, children={})
        child = SimpleNamespace(
            graph=object() if captured else None, children={1: [leaf]}
        )
        root = SimpleNamespace(graph=None, children={0: [child]})
        manager = SimpleNamespace(get_roots=lambda: iter([root]))
        module = SimpleNamespace(get_manager=lambda *args, **kwargs: manager)
        values = {
            "stats": {"unique_graphs": 1},
            "inductor": {"cudagraph_skips": int(case == "skipped")},
        }
        with (
            mock.patch.dict(counters, values, clear=True),
            mock.patch.dict(sys.modules, {"torch._inductor.cudagraph_trees": module}),
        ):
            if case == "captured":
                diagnostics = benchmark._compiler_diagnostics("full", "cuda:0", True)
                self.assertEqual(diagnostics["cudagraph_capture_count"], 2)
            else:
                message = "skipped them 1 time" if captured else "no graph capture"
                with self.assertRaisesRegex(RuntimeError, message):
                    benchmark._compiler_diagnostics("full", "cuda:0", True)

    def test_hierarchical_requires_nested_region(self):
        from torch._dynamo.utils import compilation_time_metrics, counters

        with (
            mock.patch.dict(counters, {"stats": {"unique_graphs": 1}}, clear=True),
            mock.patch.dict(compilation_time_metrics, {}, clear=True),
        ):
            with self.assertRaisesRegex(RuntimeError, "no nested region"):
                benchmark._compiler_diagnostics("hierarchical", "cpu", False)
            compilation_time_metrics["invoke_subgraph_trace"] = [0.1]
            benchmark._compiler_diagnostics("hierarchical", "cpu", False)

    def test_extract_tensor_handles_pipeline_outputs(self):
        import numpy as np
        from PIL import Image

        pixels = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
        expected = torch.from_numpy(pixels).expand(2, 2, 2, 3)
        fields = ("images", "frames")
        for output in (
            {"images": [Image.fromarray(pixels)] * 2},
            SimpleNamespace(frames=np.stack([pixels, pixels])),
        ):
            self.assertEqual(benchmark._extract_tensor(output, fields), expected)
        stacked = benchmark._extract_tensor((torch.ones(3), torch.ones(3)), fields)
        self.assertEqual(stacked, torch.ones(2, 3))
        with self.assertRaisesRegex(TypeError, "cannot extract a tensor"):
            benchmark._extract_tensor(object(), fields)

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
