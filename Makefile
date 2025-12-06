# This makefile does nothing but delegating the actual building to cmake.

SHELL        = /bin/bash
.SHELLFLAGS := -eu -o pipefail -c
PYTHON      ?= $(shell command -v python3 || command -v python)
PIP          = $(PYTHON) -m pip
NIGHTLY_TOOL_OPTS := pull

.PHONY: all
all:
	@cmake -S . -B build $(shell $(PYTHON) ./scripts/get_python_cmake_flags.py) && \
		cmake --build build --parallel --

.PHONY: local
local:
	@./scripts/build_local.sh

.PHONY: android
android:
	@./scripts/build_android.sh

.PHONY: ios
ios:
	@./scripts/build_ios.sh

.PHONY: triton
triton:
	$(PIP) uninstall -y triton
	@./scripts/install_triton_wheel.sh

.PHONY: clean
clean: # This will remove ALL build folders.
	@rm -r build*/ || true

.PHONY: linecount
linecount:
	@cloc --read-lang-def=caffe.cloc caffe2 || \
		echo "Cloc is not available on the machine. You can install cloc with " && \
		echo "    sudo apt-get install cloc"

.PHONY: ensure-branch-clean
ensure-branch-clean:
	@if [ -n "$(shell git status --porcelain)" ]; then \
		echo "Please commit or stash all changes before running this script"; \
		exit 1; \
	fi

.PHONY: setup-env
setup-env: ensure-branch-clean
	$(PYTHON) tools/nightly.py $(NIGHTLY_TOOL_OPTS)

.PHONY: setup-env-cuda
setup-env-cuda:
	$(MAKE) setup-env PYTHON="$(PYTHON)" NIGHTLY_TOOL_OPTS="$(NIGHTLY_TOOL_OPTS) --cuda"

.PHONY: setup-env-rocm
setup-env-rocm:
	$(MAKE) setup-env PYTHON="$(PYTHON)" NIGHTLY_TOOL_OPTS="$(NIGHTLY_TOOL_OPTS) --rocm"

.PHONY: setup-lint
setup-lint .lintbin/.lintrunner.sha256: requirements.txt pyproject.toml .lintrunner.toml
	@echo "Setting up lintrunner..."
	$(PIP) install lintrunner
	lintrunner init
	@echo "Generating .lintrunner.sha256..."
	@mkdir -p .lintbin
	@sha256sum requirements.txt pyproject.toml .lintrunner.toml > .lintbin/.lintrunner.sha256

.PHONY: lazy-setup-lint
lazy-setup-lint: .lintbin/.lintrunner.sha256
	@if [ ! -x "$(shell command -v lintrunner)" ]; then \
		$(MAKE) setup-lint; \
	fi

.PHONY: lint
lint: lazy-setup-lint
	lintrunner --all-files

.PHONY: quicklint
quicklint: lazy-setup-lint
	lintrunner

.PHONY: quickfix
quickfix: lazy-setup-lint
	lintrunner --apply-patches

# Deprecated target aliases
.PHONY: setup_env setup_env_cuda setup_env_rocm setup_lint
setup_env: setup-env
setup_env_cuda: setup-env-cuda
setup_env_rocm: setup-env-rocm
setup_lint: setup-lint
