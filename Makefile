# This makefile does nothing but delegating the actual building to cmake.
PYTHON = python3
PIP = pip3

all:
	@mkdir -p build && cd build && cmake .. $(shell $(PYTHON) ./scripts/get_python_cmake_flags.py) && $(MAKE)

local:
	@./scripts/build_local.sh

android:
	@./scripts/build_android.sh

ios:
	@./scripts/build_ios.sh

clean: # This will remove ALL build folders.
	@rm -r build*/

linecount:
	@cloc --read-lang-def=caffe.cloc caffe2 || \
		echo "Cloc is not available on the machine. You can install cloc with " && \
		echo "    sudo apt-get install cloc"

shellcheck:
	@$(PYTHON) tools/actions_local_runner.py \
		--file .github/workflows/lint.yml \
		--job 'workflow-checks' \
		--step "Regenerate workflows"
	@$(PYTHON) tools/actions_local_runner.py \
		--file .github/workflows/lint.yml \
		--job 'workflow-checks' \
		--step "Assert that regenerating the workflows didn't change them"

setup_lint:
	$(PIP) install lintrunner
	lintrunner init
	$(PYTHON) -mpip install jinja2 --user

quick_checks:
# TODO: This is broken when 'git config submodule.recurse' is 'true' since the
# lints will descend into third_party submodules
	@$(PYTHON) tools/actions_local_runner.py \
		--file .github/workflows/lint.yml \
		--job 'quick-checks' \
		--step 'Ensure no versionless Python shebangs'

toc:
	@$(PYTHON) tools/actions_local_runner.py \
		--file .github/workflows/lint.yml \
		--job 'toc' \
		--step "Regenerate ToCs and check that they didn't change"

lint: quick_checks shellcheck
	lintrunner

quicklint: CHANGED_ONLY=--changed-only
quicklint: quick_checks shellcheck
	lintrunner
