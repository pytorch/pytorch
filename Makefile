# This makefile does nothing but delegating the actual building to cmake.
PYTHON = python3

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
		--job 'shellcheck'

setup_lint:
	$(PYTHON) tools/actions_local_runner.py --file .github/workflows/lint.yml \
		--job 'flake8-py3' --step 'Install dependencies' --no-quiet
	$(PYTHON) tools/actions_local_runner.py --file .github/workflows/lint.yml \
		--job 'cmakelint' --step 'Install dependencies' --no-quiet
	$(PYTHON) tools/actions_local_runner.py --file .github/workflows/lint.yml \
		--job 'mypy' --step 'Install dependencies' --no-quiet
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
	pip install jinja2
	pip install -r tools/linter/clang_tidy/requirements.txt
	$(PYTHON) -m tools.linter.install.clang_tidy

quick_checks:
# TODO: This is broken when 'git config submodule.recurse' is 'true' since the
# lints will descend into third_party submodules
	@$(PYTHON) tools/actions_local_runner.py \
		--file .github/workflows/lint.yml \
		--job 'quick-checks' \
		--step 'Ensure no trailing spaces' \
		--step 'Ensure no tabs' \
		--step 'Ensure no non-breaking spaces' \
		--step 'Ensure canonical include' \
		--step 'Ensure no versionless Python shebangs' \
		--step 'Ensure no unqualified noqa' \
		--step 'Ensure no unqualified type ignore' \
		--step 'Ensure no direct cub include' \
		--step 'Ensure correct trailing newlines'

flake8:
	@$(PYTHON) tools/actions_local_runner.py \
		$(CHANGED_ONLY) \
		--job 'flake8-py3'

mypy:
	@$(PYTHON) tools/actions_local_runner.py \
		$(CHANGED_ONLY) \
		--job 'mypy'

cmakelint:
	@$(PYTHON) tools/actions_local_runner.py \
		--file .github/workflows/lint.yml \
		--job 'cmakelint' \
		--step 'Run cmakelint'

clang-tidy:
	@$(PYTHON) tools/actions_local_runner.py \
		$(CHANGED_ONLY) \
		--job 'clang-tidy'

toc:
	@$(PYTHON) tools/actions_local_runner.py \
		--file .github/workflows/lint.yml \
		--job 'toc' \
		--step "Regenerate ToCs and check that they didn't change"

lint: flake8 mypy quick_checks cmakelint shellcheck

quicklint: CHANGED_ONLY=--changed-only
quicklint: mypy flake8 quick_checks cmakelint shellcheck clang-tidy
