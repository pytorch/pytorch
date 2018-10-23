#include "label_gain_map_op.h"

#include "caffe2/core/context.h"

namespace caffe2 {
namespace {

const std::string doc = R"DOC(
Lookup a value in a map by key in the batch mode.
If the key is not in the map, the returned value is the interpolation if possible.

Example input:
keys = [[11, 22, 0], [12, 0, 31]]

Arguments to construct the maps:
map_lengths = [2, 3, 2]
map_keys = [11, 12, 21, 22, 23, 31, 32]
map_values = [11.0, 12.0, 21.0, 22.0, 23.0, 31.0, 32.0]

Example output:
values = [[11.0, 22.0, 0.0], [12.0, 0.0, 31.0]]
presence = [[True, True, False], [True, False, True]]
)DOC";

REGISTER_CPU_OPERATOR(LabelGainMap, LabelGainMapOp<CPUContext>);
NO_GRADIENT(LabelGainMap);
OPERATOR_SCHEMA(LabelGainMap)
    .SetDoc(doc)
    .NumInputs(1)
    .NumOutputs(1)
    .Arg("map_lengths", "The length of each provided map.")
    .Arg("map_keys", "The map keys.")
    .Arg("map_values", "The map values.")
    .Input(0, "keys", "The input keys")
    .Output(0, "values", "The output mapped keys, with the same size as keys");
} // namespace
} // namespace caffe2
