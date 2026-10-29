"""Full native parent/child controls for core LLVM or AMD codegen SDK updates."""

import json
import sys

import triton_pin_mi350_controls as controls


def main():
    amd = "--amd" in sys.argv
    suffix = "amd-boundary" if amd else "boundary"
    controls.REPORT = controls.ROOT / ("test/test-reports/triton-pin-mi350-" + suffix)
    controls.WORK = controls.ROOT / ("build/triton-pin-mi350-" + suffix)
    controls.REPORT.mkdir(parents=True, exist_ok=True)
    controls.WORK.mkdir(parents=True, exist_ok=True)
    controls.run(
        [
            sys.executable,
            "-c",
            "import torch; p=torch.cuda.get_device_properties(0); "
            "assert torch.version.hip and 'gfx950' in p.gcnArchName, str(p); print(p)",
        ],
        cwd=controls.ROOT / "test",
        name="mi350-preflight",
        timeout=180,
    )
    revisions = {
        "before-core-llvm": "64ccfff9e9bd39c5d7136d904e4671855cbe0438",
        "after-core-llvm": "1e259627c3426df5372396ba0640b76f3deb6402",
    }
    if amd:
        revisions = {
            "before-amd-codegen": "d364bf19639a3f4a8aecb45bca0a533e2ecb6258",
            "after-amd-codegen": "ad50bab56a7ded8442f985c741449d5d3e35ae59",
        }
    results = {}
    for label, sha in revisions.items():
        site = controls.build_pin(
            label, "https://github.com/triton-lang/triton.git", sha
        )
        results[label] = controls.test_pin(label, site, sha)
        (controls.REPORT / "results.json").write_text(json.dumps(results, indent=2))
    print(
        "Full native LLVM SDK parent/child controls completed",
        json.dumps(results),
        flush=True,
    )


if __name__ == "__main__":
    main()
