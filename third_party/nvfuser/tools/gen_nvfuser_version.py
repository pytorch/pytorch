import subprocess
from pathlib import Path

UNKNOWN = "Unknown"

# note that this root currently is still part of pytorch.
def get_sha(nvfuser_root: Path) -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=nvfuser_root)
            .decode("ascii")
            .strip()
        )
    except Exception:
        return UNKNOWN

if __name__ == "__main__":
    nvfuser_root = Path(__file__).parent.parent
    version_file = nvfuser_root / "python" / "version.py"
    sha = get_sha(nvfuser_root)
    version = open((nvfuser_root / "version.txt"), "r").read().strip() + "+git" + sha[:7]
    with open(version_file, "w") as f:
        f.write("_version_str = '{}'\n".format(version))
