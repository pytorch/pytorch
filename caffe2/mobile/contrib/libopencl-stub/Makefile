CC = gcc
RANLIB = ranlib

LIBSRC = src/libopencl.c
LIBOBJ=$(LIBSRC:.c=.o)

CFLAGS = -O2 -fPIC -I ./include -Wall

LIBOPENCL = libOpenCL.a
TARGETS = $(LIBOPENCL)

all: $(TARGETS)

libopencl.o: libopencl.c
	$(CC) $(CFLAGS) -c libopencl.c -o libopencl.o

$(TARGETS): $(LIBOBJ)
	ar rcs $(LIBOPENCL) src/libopencl.o
	$(RANLIB) $(LIBOPENCL)

clean:
	rm -f $(TARGETS) $(LIBOBJ)

