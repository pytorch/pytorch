# This makefile does nothing but delegating the actual compilation to build.py.

all:
	@mkdir -p build && cd build && cmake .. && make

#android:
#	@python brewtool/build_android.py build

clean:
	@rm -r build/

#test:
#	@python brewtool/build.py test

#lint:
#	@find caffe2 -type f -exec python brewtool/cpplint.py {} \;

linecount:
	@cloc --read-lang-def=caffe.cloc caffe2 || \
		echo "Cloc is not available on the machine. You can install cloc with " && \
		echo "    sudo apt-get install cloc"
