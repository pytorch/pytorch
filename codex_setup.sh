uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
#COMMIT=$(curl -s https://api.github.com/repos/pytorch/pytorch/branches/nightly | jq ".commit.commit.message" | sed -E 's/.*\(([0-9a-f]{40})\).*/\1/')
git reset --hard 74a754aae98aabc2aca67e5edb41cc684fae9a82
curl https://patch-diff.githubusercontent.com/raw/pytorch/pytorch/pull/159965.diff | patch -p1
USE_NIGHTLY=2.9.0.dev20250806+cpu python setup.py develop
git reset --hard HEAD
echo "source .venv/bin/activate" >> ~/.bashrc
