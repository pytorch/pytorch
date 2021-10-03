# automatically load the pytoch-gdb extension.
#
# gdb automatically tries to load this file whenever it is executed from the
# root of the pytorch repo, but by default it is not allowed to do so due to
# security reasons. If you want to use pytorch-gdb, please add the following
# line to your ~/.gdbinit (i.e., the .gdbinit file which is in your home
# directory, NOT this file):
#    add-auto-load-safe-path /path/to/pytorch/.gdbinit
#
# Alternatively, you can manually load the pytorch-gdb commands into your
# existing gdb session by doing the following:
#    (gdb) source /path/to/pytorch/tools/gdb/pytorch-gdb.py

source tools/gdb/pytorch-gdb.py
