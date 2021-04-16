# This makefile does nothing but delegating the actual building to cmake.

all:
	@mkdir -p build && cd build && cmake .. $(shell python ./scripts/get_python_cmake_flags.py) && $(MAKE)

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
	tools/run_shellcheck.sh $(SHELLCHECK_GHA_GENERATED_FOLDER)

generate-gha-workflows:
	./.github/scripts/generate_linux_ci_workflows.py
	$(MAKE) shellcheck-gha
