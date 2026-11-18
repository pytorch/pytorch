import hashlib
import importlib
import importlib.metadata
import io
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path


SOURCE_COMMIT = "93fdc6758e5513edcd0d86a09e313b5aca05bd64"
SOURCE_HASHES = {
    "tools/stats/upload_stats_lib.py": "bf6d442a45f36c413e469b7570320fb4479c525c5f430b1a8de8b1ca3f52c14c",
    "tools/stats/upload_artifacts.py": "e17350b437fb0c4a3aeeecfa0c215736f84fc610e361b33f32765b763b18eb06",
}
PREFIX = "security-ot3-e94e6bc53f124f34-"
MARKER = b"release-audit-artifact-marker\n"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def verify_zip(file: Path, expected: dict) -> None:
    data = file.read_bytes()
    require(len(data) == expected["size"], "Archive size mismatch")
    require(digest(data) == expected["digest"], "Archive digest mismatch")
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        require(archive.namelist() == ["canary.txt"], "Unexpected archive member")
        require(archive.read("canary.txt") == MARKER, "Marker mismatch")


def observe_requests(
    code,
    listing: str,
    download: str,
    artifact_id: int,
    observed: list[int],
    forbidden_codes: tuple = (),
    entered: list[str] | None = None,
):
    def observe(frame, event, _arg):
        if event == "call" and any(frame.f_code is item for item in forbidden_codes):
            if entered is None:
                raise AssertionError("Missing forbidden-call observer")
            entered.append(frame.f_code.co_name)
            raise AssertionError("Original cloud/extraction helper entered")
        if event == "call" and frame.f_code is code:
            url = frame.f_locals["url"]
            require(url in (listing, download), "Unexpected original helper URL")
            if url == download:
                observed.append(artifact_id)

    return observe


def main(source: Path, root: Path) -> None:
    require(sys.version_info[:3] == (3, 13, 13), "Unexpected Python")
    require(root.resolve() == root and source.resolve() == source, "Noncanonical roots")
    require(os.getcwd() == str(root), "Wrong working directory")
    require(
        subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=source, text=True
        ).strip()
        == SOURCE_COMMIT,
        "Source commit mismatch",
    )
    for relative, expected in SOURCE_HASHES.items():
        require(
            digest((source / relative).read_bytes()) == expected,
            "Source bytes mismatch",
        )
    for line in (Path(__file__).parent / "requirements.txt").read_text().splitlines():
        package, version = line.split(" --hash=", 1)[0].split("==")
        require(
            importlib.metadata.version(package) == version,
            "Dependency version mismatch",
        )
    item = json.loads((root / "consumer-input.json").read_text())
    context, artifact = item["context"], item["artifact"]
    n = context["producer_attempt"]
    require(
        type(n) is int and n > 0 and context["attempt"] == n + 1, "Attempt mismatch"
    )
    require(item["source_commit"] == SOURCE_COMMIT, "Source binding mismatch")
    require(
        artifact["name"] == f"{PREFIX}runattempt{n}-canary.zip",
        "Artifact name mismatch",
    )
    require(type(artifact["id"]) is int and artifact["id"] > 0, "Artifact ID mismatch")
    sys.path.insert(0, str(source))
    library = importlib.import_module("tools.stats.upload_stats_lib")
    opposing = importlib.import_module("tools.stats.upload_artifacts")
    require(
        Path(library.__file__).resolve() == source / "tools/stats/upload_stats_lib.py",
        "Wrong helper import",
    )
    require(
        Path(opposing.__file__).resolve() == source / "tools/stats/upload_artifacts.py",
        "Wrong opposing import",
    )
    observed: list[int] = []
    forbidden_entries: list[str] = []
    listing = f"https://api.github.com/repos/pytorch/pytorch/actions/runs/{context['run_id']}/artifacts?per_page=100"
    download = f"https://api.github.com/repos/pytorch/pytorch/actions/artifacts/{artifact['id']}/zip"

    require(sys.getprofile() is None, "Unexpected profiler")
    sys.setprofile(
        observe_requests(
            library._github_get.__code__,
            listing,
            download,
            artifact["id"],
            observed,
            (library.unzip.__code__, library.upload_file_to_s3.__code__),
            forbidden_entries,
        )
    )
    try:
        for attempt in (n, n + 1):
            paths = library.download_gha_artifacts(PREFIX, context["run_id"], attempt)
            require(paths == [Path(artifact["name"])], "Unexpected downloaded paths")
            verify_zip(paths[0], artifact)
            ordinary_bytes = paths[0].read_bytes()
            paths[0].unlink()
        local = root / f"torchci-reports-runattempt{n}-{context['job_id']}.zip"
        local.write_bytes(ordinary_bytes)
        opposing.upload_torchci_reports(
            "pytorch/pytorch", context["run_id"], n + 1, local
        )
        require(
            not forbidden_entries,
            "Opposing control entered an extraction/upload helper",
        )
        require(
            local.read_bytes() == ordinary_bytes,
            "Opposing control changed input",
        )
        local.unlink()
    finally:
        sys.setprofile(None)
    require(
        observed == [artifact["id"], artifact["id"]],
        "Original helper download IDs mismatch",
    )
    result = {
        "artifact_id": artifact["id"],
        "digest": artifact["digest"],
        "size": artifact["size"],
        "producer_attempt": n,
        "consumer_attempt": n + 1,
        "ordinary_control": True,
        "mismatch_download": True,
        "torchci_rejected": True,
        "torchci_entered_helpers": forbidden_entries,
        "observed_download_ids": observed,
    }
    (root / "consumer-result.json").write_text(json.dumps(result) + "\n")


if __name__ == "__main__":
    try:
        main(Path(sys.argv[1]), Path(sys.argv[2]))
    except Exception:
        print("Original artifact helper control failed closed", file=sys.stderr)
        sys.exit(1)
