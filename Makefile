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

setup_lint:
	$(PIP) install lintrunner
	lintrunner init

lint:
	lintrunner

quicklint:
	lintrunner

triton:
	$(PIP) uninstall -y triton
	@./scripts/install_triton_wheel.sh
