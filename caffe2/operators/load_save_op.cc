#include "caffe2/operators/load_save_op.h"

namespace caffe2 {

template <>
void LoadOp<CPUContext>::SetCurrentDevice(BlobProto* proto) {
  if (proto->has_tensor()) {
    proto->mutable_tensor()->clear_device_detail();
    proto->mutable_tensor()->mutable_device_detail()->set_device_type(
        PROTO_CPU);
  }
}

template <int VALUE_TYPE = TensorProto_DataType_FLOAT>
std::vector<TensorShape> LoadTensorInference(
    const OperatorDef& def,
    const vector<TensorShape>& /* unused */) {
  ArgumentHelper helper(def);
  auto shape = helper.GetRepeatedArgument<int64_t>("shape");
  vector<TensorShape> out;
  // Currently load op supports only shape.
  // TODO: We have to extend it to support shapes vector.
  // Since it support just one shape, we return
  // the right shape information only when there is just one blob loaded.
  // Otherwise, we return unknown TensorShapes.
  if (def.output_size() == 1 && shape.size() > 0) {
    TensorShape ts;
    ts.set_data_type(static_cast<TensorProto_DataType>(
        helper.GetSingleArgument<int>("dtype", VALUE_TYPE)));
    for (auto d : shape) {
      ts.add_dims(d);
    }
    out.push_back(ts);
  } else {
    for (int i = 0; i < def.output_size(); i++) {
      TensorShape ts;
      ts.set_unknown_shape(true);
      out.push_back(ts);
    }
  }
  return out;
}

REGISTER_CPU_OPERATOR(DBExists, DBExistsOp<CPUContext>);
REGISTER_CPU_OPERATOR(Load, LoadOp<CPUContext>);
REGISTER_CPU_OPERATOR(Save, SaveOp<CPUContext>);
REGISTER_CPU_OPERATOR(Checkpoint, CheckpointOp<CPUContext>);
// CPU Operator old name: do NOT use, we may deprecate this later.
REGISTER_CPU_OPERATOR(Snapshot, CheckpointOp<CPUContext>);

OPERATOR_SCHEMA(DBExists)
    .NumInputs(0)
    .NumOutputs(1)
    .SetDoc(R"DOC(
Checks if the db described by the arguments exists.

Github Links:

- https://github.com/pytorch/pytorch/blob/master/caffe2/operators/load_save_op.cc

<details>

<summary> <b>Example</b> </summary>

**Code**

```
workspace.ResetWorkspace()

op = core.CreateOperator(
    "DBExists",
    [],
    ["exists"],
    db_name="test_db",
    db_type="leveldb",
)

workspace.RunOperatorOnce(op)
print("exists:", workspace.FetchBlob("exists"))

```

</details>

)DOC")
    .Output(0, "exists", "*(type: Tensor`<bool>`)* Scalar boolean output "
    "tensor. True if the db exists, else false.")
    .Arg(
        "absolute_path",
        "*(type: int; default: 0)* If set to non-zero, save the db directly to "
        "the path specified by the `db` arg. If not set (default), prepend the "
        "path of the current root folder of the workspace to the path specified "
        "by the `db` arg.")
    .Arg("db_name", "*(type: string)* Path to the db in question; see the "
    "`absolute_path` arg details for options regarding the current root folder "
    "of the workspace.")
    .Arg("db_type", "*(type: string)* Type of db to save (options: \"lmdb\", "
    "\"leveldb\", \"minidb\").");

OPERATOR_SCHEMA(Load)
    .NumInputs(0, INT_MAX)
    .NumOutputs(0, INT_MAX)
    .TensorInferenceFunction(LoadTensorInference<>)
    .SetDoc(R"DOC(
The Load operator loads a set of serialized blobs from a db or multiple dbs. It
takes $[0, \infty)$ number of inputs and $[0, \infty)$ number of outputs, using
the db keys to match the db entries with the outputs.

If at least one input is passed, then it is assumed that that input blobs are a
set of DBReaders to load from. Otherwise the `db` or `dbs` argument is used to load
blobs from one single db or multiple dbs respectively. `db_type` argument is used
to specify the type of the input db/dbs.

Github Links:

- https://github.com/pytorch/pytorch/blob/master/caffe2/operators/load_save_op.cc

<details>

<summary> <b>Example</b> </summary>

**Code**

```
workspace.ResetWorkspace()

op = core.CreateOperator(
    "Load",
    [],
    ["X", "Y"],
    db="test_db",
    db_type="lmdb"
)

workspace.RunOperatorOnce(op)
print("X:", workspace.FetchBlob("X"))
print("Y:", workspace.FetchBlob("Y"))

```

</details>

)DOC")
    .Input(
      0,
      "X, Y, ...",
      "*(type: List(DBReader))* [OPTIONAL] List of DBReaders to load from. Can "
      "use this instead of the `db`/`dbs` args.")
    .Arg(
        "absolute_path",
        "*(type: int; default: 0)* If set to non-zero, save the db directly to "
        "the path specified by the `db` arg. If not set (default), prepend the "
        "path of the current root folder of the workspace to the path specified "
        "by the `db` arg.")
    .Arg(
        "add_prefix",
        "*(type: string, default: \"\")* Blobs will be prefixed with this when "
        "loading. Useful for avoiding collisions with blobs existing in the "
        "workspace. The output blob names specified to this op should include "
        "this prefix.")
    .Arg(
        "strip_prefix",
        "*(type: string, default: \"\")* Characters in the provided blob names "
        "that match `strip_prefix` will be removed prior to saving. Also, "
        "characters that precede `strip_prefix` will be removed. Useful for "
        "removing device scope from blob names.")
    .Arg("db", "*(type: string)* The output path of the db. See the "
        "`absolute_path` arg details for options regarding the current root folder "
        "of the workspace.")
    .Arg(
        "dbs",
        "*(type: List(string))* List of paths to dbs to load blobs from. See "
        "the `absolute_path` arg details for options regarding the current "
        "root folder of the workspace.")
    .Arg("db_type", "(type: string)* Type of db to save (options: \"lmdb\", "
        "\"leveldb\", \"minidb\").")
    .Arg(
        "keep_device",
        "*(type: int; default: 0)* If nonzero, the blobs are loaded into the "
        "device that is specified in the serialized `BlobProto`. Otherwise, "
        "the device will be set as the one that the `Load` operator is being "
        "run under.")
    .Arg(
        "load_all",
        "*(type: int; default: 0)* If nonzero, will load all blobs pointed to "
        "by the db to the workspace overwriting/creating blobs as needed.")
    .Arg(
        "allow_incomplete",
        "*(type: bool; default: False)* If True, will allow not loading all "
        "the output blobs specified in the outputs.")
    .Arg(
        "source_blob_names",
        "*(type: List(string))* If set, used instead of output blob names to "
        "specify which blobs in the db shall be loaded. Must be the same "
        "length as number of output blobs.");

OPERATOR_SCHEMA(Save)
    .NumInputs(1, INT_MAX)
    .NumOutputs(0)
    .SetDoc(R"DOC(
Saves a set of blobs to a db. It takes $[1, \infty)$ number of inputs and has
no output. The contents of the inputs are written into the db using the
settings specified by the arguments.

Github Links:

- https://github.com/pytorch/pytorch/blob/master/caffe2/operators/load_save_op.cc

<details>

<summary> <b>Example</b> </summary>

**Code**

```
workspace.ResetWorkspace()

op = core.CreateOperator(
    "Save",
    ["X", "Y", "Z"],
    [],
    db="test_db2",
    db_type="leveldb",
    blob_name_overrides=["x_scores", "y_scores", "z_scores"]
)

workspace.FeedBlob("X", np.random.randint(20, size=(5,5)))
workspace.FeedBlob("Y", np.random.randint(20, size=(5,5)))
workspace.FeedBlob("Z", np.random.randint(20, size=(5,5)))
workspace.RunOperatorOnce(op)

```

</details>

)DOC")
    .Arg(
        "absolute_path",
        "*(type: int; default: 0)* If set to non-zero, save the db directly to "
        "the path specified by the `db` arg. If not set (default), prepend the "
        "path of the current root folder of the workspace to the path specified "
        "by the `db` arg.")
     .Arg(
         "strip_prefix",
         "*(type: string, default: \"\")* Characters in the provided blob names "
         "that match `strip_prefix` will be removed prior to saving. Also, "
         "characters that precede `strip_prefix` will be removed. Useful for "
         "removing device scope from blob names.")
    .Arg(
        "blob_name_overrides",
        "*(List(string))* If set, used as blob names instead of original blob "
        "names. Must be same length as number of blobs.")
    .Arg("db", "*(type: string)* The output path of the db. See the "
    "`absolute_path` arg details for options regarding the current root folder "
    "of the workspace.")
    .Arg("db_type", "*(type: string)* Type of db to save (options: \"lmdb\", "
    "\"leveldb\", \"minidb\").")
    .Arg("chunk_size", "*(type: string; default: kDefaultChunkSize)* The chunk "
    "size to split tensor data into. If not set, caffe2_tensor_chunk_size will "
    "be used")
    .Input(0, "X", "*(type: Tensor)* Input tensor(s).");

OPERATOR_SCHEMA(Checkpoint)
    .NumInputs(1, INT_MAX)
    .NumOutputs(0)
    .SetDoc(R"DOC(
The Checkpoint operator is similar to the Save operator, but allows one to save
to db every few iterations, with a db name that is appended with the iteration
count. It takes [1, infinity) number of inputs and has no output. The first
input has to be a TensorCPU of type int and has size 1 (i.e. the iteration
counter). This is determined whether we need to do checkpointing.
)DOC")
    .Arg(
        "absolute_path",
        "(int, default 0) if set, use the db path directly and do not prepend "
        "the current root folder of the workspace.")
    .Arg(
        "db",
        "(string) a template string that one can combine with the "
        "iteration to create the final db name. For example, "
        "\"/home/lonestarr/checkpoint_%08d.db\"")
    .Arg("db_type", "(string) the type of the db.")
    .Arg(
        "every",
        "(int, default 1) the checkpointing is carried out when "
        "(iter mod every) is zero.");

OPERATOR_SCHEMA(Snapshot);

NO_GRADIENT(Load);
SHOULD_NOT_DO_GRADIENT(DBExists);
SHOULD_NOT_DO_GRADIENT(Save);
SHOULD_NOT_DO_GRADIENT(Checkpoint);
SHOULD_NOT_DO_GRADIENT(Snapshot);
}  // namespace caffe2
