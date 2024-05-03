import json
import glob
from pathlib import Path
import re
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent


if __name__ == "__main__":
    all_labels = set()
    keywords = set(["quantized", "mps", "sparse"])
    for file in glob.glob(f"{REPO_ROOT}/test/**/*.py", recursive=True):
        rel_path = Path(file).relative_to(REPO_ROOT / "test")

        for folder in rel_path.parts[:-1]:
            keywords.add(folder)

        with open(file) as f:
            for line in f.readlines():
                rem = re.search(r"Owner\(s\): (\[.*\])", line)
                if rem:
                    all_labels.update(json.loads(rem.groups()[0]))
    # print(json.dumps(list(all_labels), indent=2))
    print(json.dumps(list(keywords), indent=2))
    totla = 0
    tta = 0
    for file in glob.glob(f"{REPO_ROOT}/**/*", recursive=True):
        # if Path(file).is_dir():
        #     continue
        rel_path = Path(file).relative_to(REPO_ROOT)
        dont_care = [
            ".ci/", "tools/", "test/", "docs/", "third_party/", "cmake/"
        ]
        if any([str(rel_path).startswith(dc) for dc in dont_care]):
            continue

        folders = []
        for folder in rel_path.parts[:-1]:
            if folder.startswith("_"):
                folders.append(folder[1:])
            else:
                folders.append(folder)
        if not any([kw in folders for kw in keywords]):
            print(rel_path)
            tta += 1
        totla += 1
    print(totla, tta, tta/totla)
