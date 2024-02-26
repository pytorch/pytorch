# Run include-what-you-use on a file or folder
# e.g. tools/iwyu/run.sh aten/src/ATen/native/sparse/SparseBlas.cpp
# Which will print suggested changes to the console
#
# Currently the include mappings aren't good enough to trust iwyu's
# output e.g. we probably just want to include Tensor.h and trust it
# brings in the c10 headers. So, for now, use iwyu as a guide and
# update includes manually.

TORCH_ROOT=$(dirname $(dirname $(dirname $(readlink -f $0))))

iwyu_tool -p $TORCH_ROOT/build $@ -- -Wno-unknown-warning-option -Xiwyu \
          --no_fwd_decls -Xiwyu --mapping_file=$TORCH_ROOT/tools/iwyu/all.imp \
    | python $TORCH_ROOT/tools/iwyu/fixup.py
