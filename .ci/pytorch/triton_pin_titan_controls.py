"""Run the original strict two-A10G TorchTitan SFT comparison with full native pins."""

import json
import os
import sys

import triton_pin_mi350_controls as controls


def main():
    titan = controls.ROOT / "torchtitan"
    controls.REPORT = controls.ROOT / "test/test-reports/triton-pin-titan-controls"
    controls.WORK = controls.ROOT / "build/triton-pin-titan-controls"
    controls.REPORT.mkdir(parents=True, exist_ok=True)
    controls.WORK.mkdir(parents=True, exist_ok=True)
    controls.run([sys.executable, "-c", "import torch; assert torch.cuda.device_count()>=2; "
                  "assert 'A10G' in torch.cuda.get_device_name(0); assert torch.version.cuda=='13.2'; "
                  "print(torch.version.git_version,torch.cuda.get_device_properties(0))"],
                 cwd=titan, name="titan-preflight", timeout=180)
    results = {}
    metrics = {}
    selected_pins = {'candidate': controls.PINS['candidate']} if '--fusion-only' in sys.argv else controls.PINS
    if '--boundary' in sys.argv:
        selected_pins = {
            'parent': ('https://github.com/triton-lang/triton.git', '7e013d6897e2037a73e8ddfe1aa2d8693d4bf349'),
            'child': ('https://github.com/triton-lang/triton.git', '1d503a35275625d2c236605a42855bfd81267047'),
        }
    for label, (repository, sha) in selected_pins.items():
        site = controls.build_pin(label, repository, sha)
        phases = [(label, None)] + ([(label + "-implicit-fusion-restored", "1")] if label in {'candidate', 'child'} else [])
        for phase, fma in phases:
            env = os.environ.copy()
            env["PYTHONPATH"] = str(site) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
            env["TRITON_CACHE_DIR"] = str(controls.WORK / phase / "triton-cache")
            env["TORCHINDUCTOR_CACHE_DIR"] = str(controls.WORK / phase / "inductor-cache")
            controls.configure_fusion(env, fma)
            env['TORCHINDUCTOR_FORCE_DISABLE_CACHES'] = '1'
            env["CUDA_VISIBLE_DEVICES"] = "0,1"
            probe = ("import triton,importlib.metadata,pathlib,torch; v=importlib.metadata.version('triton'); "
                     f"assert '{sha[:8]}' in v,v; assert pathlib.Path(triton.__file__).resolve().is_relative_to({str(site)!r}); "
                     + controls.fusion_provenance(fma)
                     + "print(v,triton.__file__,torch.version.git_version)")
            controls.run([sys.executable, "-c", probe], cwd=titan, env=env, name=phase + "-provenance", timeout=180)
            results[phase] = controls.run([sys.executable, "-m", "tests.integration_tests.run_tests",
                         "--test_suite", "features", "--test_name", "sft", "--ngpu", "2",
                         str(controls.REPORT / phase)], cwd=titan, env=env,
                         name=phase + "-original-sft", check=False, timeout=1800)
            log = (controls.REPORT / (phase + "-original-sft.log")).read_text(errors="replace")
            assert results[phase] in (0, 1) and ("LOSS_COMPARE" in log or "All tests passed" in log), phase
            from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

            event_root = controls.REPORT / phase / 'sft/tb_baseline'
            events = list(event_root.rglob('events.out.tfevents.*'))
            assert events, str(event_root)
            values = {}
            for path in events:
                accumulator = EventAccumulator(str(path))
                accumulator.Reload()
                for tag in ['loss_metrics/global_avg_loss', 'grad_norm']:
                    if tag in accumulator.Tags()['scalars']:
                        for value in accumulator.Scalars(tag):
                            values.setdefault(tag, {})[value.step] = value.value
            assert all(len(values.get(tag, {})) == 10 for tag in ['loss_metrics/global_avg_loss', 'grad_norm']), values
            metrics[phase] = values
            (controls.REPORT / 'metrics.json').write_text(json.dumps(metrics, indent=2))
            (controls.REPORT / "results.json").write_text(json.dumps(results, indent=2))
    print("TITAN_NATIVE_CONTROLS", json.dumps(results), flush=True)


if __name__ == "__main__":
    main()
