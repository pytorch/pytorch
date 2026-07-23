#!/bin/bash
set -ex

# Where to create the venv. Defaults to the linter image location; the main CI
# image overrides it to a per-version path.
VENV_PATH="${VENV_PATH:-/var/lib/jenkins/ci_env}"

apt-get update
# Use deadsnakes in case we need an older python version
sudo add-apt-repository -y ppa:deadsnakes/ppa

if [[ "$PYTHON_FREETHREADED" == "1" ]]; then
  PYTHON="python${PYTHON_VERSION}t"
  apt-get install -y "python${PYTHON_VERSION}-nogil" "python${PYTHON_VERSION}-dev" python3-pip "python${PYTHON_VERSION}-venv"
else
  PYTHON="python${PYTHON_VERSION}"
  apt-get install -y "python${PYTHON_VERSION}" "python${PYTHON_VERSION}-dev" python3-pip "python${PYTHON_VERSION}-venv"
fi

# Use a venv because uv and some other package managers don't support --user install
ln -sf "/usr/bin/${PYTHON}" /usr/bin/python
"${PYTHON}" -m venv "${VENV_PATH}"
source "${VENV_PATH}/bin/activate"

python -mpip install --upgrade pip
python -mpip install -r /opt/requirements-ci.txt

if [ -n "$DOCS" ]; then
  apt-get update
  apt-get -y install expect-dev
  python -mpip install -r /opt/requirements-docs.txt
fi

# Hand the venv to the runtime user so later (as_jenkins) installs can write to it
chown -R jenkins:jenkins "${VENV_PATH}"
