set -ex
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
uv pip install numpy
lintrunner init
git rev-parse HEAD > /tmp/orig_work.txt
# tools/nightly.py checks out the nightly commit (detached HEAD) and installs
# the matching pre-built nightly binaries into the repo, avoiding a
# from-source build.
python tools/nightly.py checkout -p .venv
echo "source $PWD/.venv/bin/activate" >> ~/.bashrc
