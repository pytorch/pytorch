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
	@$(RM) -r $(SHELLCHECK_GHA_GENERATED_FOLDER)

linecount:
	@cloc --read-lang-def=caffe.cloc caffe2 || \
		echo "Cloc is not available on the machine. You can install cloc with " && \
		echo "    sudo apt-get install cloc"

SHELLCHECK_GHA_GENERATED_FOLDER=.shellcheck_generated_gha
shellcheck-gha:
	@$(RM) -r $(SHELLCHECK_GHA_GENERATED_FOLDER)
	tools/extract_scripts.py --out=$(SHELLCHECK_GHA_GENERATED_FOLDER)
	tools/linter/run_shellcheck.sh $(SHELLCHECK_GHA_GENERATED_FOLDER)

generate-gha-workflows:
	.github/scripts/generate_ci_workflows.py
	$(MAKE) shellcheck-gha

shellcheck:
	@$(PYTHON) tools/actions_local_runner.py \
		--file .github/workflows/lint.yml \
		--job 'shellcheck' \
		--step "Regenerate workflows"
	@$(PYTHON) tools/actions_local_runner.py \
		--file .github/workflows/lint.yml \
		--job 'shellcheck' \
		--step "Assert that regenerating the workflows didn't change them"
	@$(PYTHON) tools/actions_local_runner.py \
		--file .github/workflows/lint.yml \
		--job 'shellcheck' \
		--step 'Extract scripts from GitHub Actions workflows'
	@$(PYTHON) tools/actions_local_runner.py \
		$(CHANGED_ONLY) \
		$(REF_BRANCH) \
		--job 'shellcheck'

setup_lint:
	$(PIP) install lintrunner
	$(PYTHON) tools/actions_local_runner.py --file .github/workflows/lint.yml \
		--job 'shellcheck' --step 'Install Jinja2' --no-quiet

	@if [ "$$(uname)" = "Darwin" ]; then \
		if [ -z "$$(which brew)" ]; then \
			echo "'brew' is required to install ShellCheck, get it here: https://brew.sh "; \
			exit 1; \
		fi; \
		brew install shellcheck; \
	else \
		$(PYTHON) tools/actions_local_runner.py --file .github/workflows/lint.yml \
		--job 'shellcheck' --step 'Install ShellCheck' --no-quiet; \
	fi
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
