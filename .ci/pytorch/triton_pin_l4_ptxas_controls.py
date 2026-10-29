"""Control the external ptxas selector with the image's matching native candidate."""

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import urllib.request

import triton
import triton_pin_mi350_controls as controls


def main():
    controls.REPORT = controls.ROOT / "test/test-reports/triton-pin-l4-ptxas"
    controls.WORK = controls.ROOT / "build/triton-pin-l4-ptxas"
    controls.REPORT.mkdir(parents=True, exist_ok=True)
    controls.WORK.mkdir(parents=True, exist_ok=True)
    debug = controls.ROOT / "test/debug/triton-pin-l4-ptxas"
    debug.mkdir(parents=True, exist_ok=True)
    version = importlib.metadata.version("triton")
    assert "1b9ee7ae" in version, version
    print("NATIVE_CANDIDATE", version, triton.__file__, flush=True)
    current_ptxas = Path(triton.__file__).parent / "backends/nvidia/bin/ptxas"
    archive = controls.WORK / "cuda_nvcc-linux-x86_64-12.9.86-archive.tar.xz"
    url = "https://developer.download.nvidia.com/compute/cuda/redist/cuda_nvcc/linux-x86_64/" + archive.name
    urllib.request.urlretrieve(url, archive)
    old_ptxas = controls.WORK / "ptxas-12.9.86"
    with tarfile.open(archive) as package:
        member = next(member for member in package.getmembers() if member.name.endswith("/bin/ptxas"))
        with package.extractfile(member) as src, old_ptxas.open("wb") as dst:
            shutil.copyfileobj(src, dst)
    old_ptxas.chmod(0o755)
    results = {}
    for label, binary in [("candidate-ptxas-13.4", current_ptxas), ("candidate-ptxas-12.9", old_ptxas)]:
        env = os.environ.copy()
        env["TRITON_PTXAS_PATH"] = str(binary)
        env["TRITON_CACHE_DIR"] = str(controls.WORK / label / "triton")
        env["TORCHINDUCTOR_CACHE_DIR"] = str(controls.WORK / label / "inductor")
        env["TORCHINDUCTOR_FORCE_DISABLE_CACHES"] = "1"
        compiler_version = subprocess.check_output([str(binary), "--version"], text=True)
        print("EXTERNAL_PTXAS", label, compiler_version, hashlib.file_digest(binary.open("rb"), "sha256").hexdigest(), flush=True)
        probe = ("import json,torch,triton; from triton.backends.nvidia.compiler import get_ptxas; "
                 "assert torch.cuda.get_device_capability(0)==(8,9); assert torch.version.cuda=='13.4'; "
                 f"assert get_ptxas(89).path=={str(binary)!r},get_ptxas(89); "
                 "x=torch.arange(65536,device='cuda',dtype=torch.int32).to(torch.int16).view(torch.bfloat16); "
                 "e=x.abs(); c=torch.compile(torch.abs,options={'numerics':'strict','triton.autotune_pointwise':False})(x); "
                 "bad=torch.nonzero(e.view(torch.int16)!=c.view(torch.int16)).flatten().cpu().tolist(); "
                 "print(json.dumps(dict(torch_sha=torch.version.git_version,device=torch.cuda.get_device_name(0),"
                 "mismatches=len(bad),examples=[dict(input=hex(i),eager=hex(e.view(torch.int16)[i].item()&65535),"
                 "compiled=hex(c.view(torch.int16)[i].item()&65535)) for i in bad[:8]]),indent=2))")
        controls.run([sys.executable, "-c", probe], cwd=controls.ROOT / "test", env=env,
                     name=label + "-all-encodings", timeout=1200)
        results[label] = controls.run([sys.executable, "-m", "pytest", "-s", "--tb=short", "--reruns=0",
                     f"--junitxml={controls.REPORT / (label + '.xml')}",
                     "inductor/test_strict_numerics.py::PointwiseStrictNumericsTestCUDA::test_pointwise_bitwise_abs_cuda_bfloat16"],
                     cwd=controls.ROOT / "test", env=env, name=label + "-original", check=False, timeout=1200)
        log = (controls.REPORT / (label + "-original.log")).read_text(errors="replace")
        assert results[label] in (0, 1) and ("1 passed" in log or "forward now matches eager under strict numerics" in log), label
        for source in (controls.WORK / label / "triton").glob("**/*"):
            if source.suffix in (".ptx", ".cubin", ".json"):
                dst = debug / label / source.relative_to(controls.WORK / label / "triton")
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, dst)
                if source.suffix == ".cubin":
                    cuobjdump = Path(triton.__file__).parent / "backends/nvidia/bin/cuobjdump"
                    sass = subprocess.check_output([str(cuobjdump), "--dump-sass", str(source)], text=True)
                    dst.with_suffix(".sass").write_text(sass)
        (controls.REPORT / "results.json").write_text(json.dumps(results, indent=2))
    print("L4_EXTERNAL_PTXAS_CONTROLS", json.dumps(results), flush=True)


if __name__ == "__main__":
    main()
