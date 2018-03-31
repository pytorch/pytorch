#include "caffe2/operators/load_save_op.h"

namespace caffe2 {

template <>
void LoadOp<CPUContext>::SetCurrentDevice(BlobProto* proto) {
  if (proto->has_tensor()) {
    proto->mutable_tensor()->mutable_device_detail()->set_device_type(CPU);
  }
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
Checks if the DB exists.
)DOC")
    .Output(0, "exists", "A scalar bool Tensor.")
    .Arg(
        "absolute_path",
        "(int, default 0) if set, use the db path directly and do not prepend "
        "the current root folder of the workspace.")
    .Arg("db_name", "(string) the path to the db to load.")
    .Arg("db_type", "(string) the type of the db.");

OPERATOR_SCHEMA(Load)
    .NumInputs(0, INT_MAX)
    .NumOutputs(0, INT_MAX)
    .SetDoc(R"DOC(
The Load operator loads a set of serialized blobs from a db or multiple dbs. It
takes [0, infinity) number of inputs and [0, infinity) number of outputs, using
the db keys to match the db entries with the outputs.

If at least one input is passed, then it is assumed that that input blobs are a
set of DBReaders to load from. Otherwise the db or dbs argument is used to load
blobs from one single db or multiple dbs respectively. db_type argument is used
to specify the type of the input db/dbs.
)DOC")
    .Arg(
        "absolute_path",
        "(int, default 0) if set, use the db path directly and do not prepend "
        "the current root folder of the workspace.")
    .Arg(
        "add_prefix",
        "(string, default=\"\") blobs will be prefixed with this when loading."
        "Useful for avoiding collisions with blobs existing in the workspace."
        "The output blob names specified to this op should include this prefix.")
    .Arg(
        "strip_prefix",
        "(string, default=\"\") characters in the provided blob "
        " names that match strip_prefix will be removed prior to loading."
        " Also, characters that precede strip_prefix will be removed. Useful "
        " for removing device scope from blob names.")
    .Arg("db", "(string) the path to the db to load.")
    .Arg(
        "dbs",
        "(list of strings) the paths to the dbs to load. This is used for loading"
        " blobs from multiple databases. If it is set, argument in \"db\" will be"
        " ignored.")
    .Arg("db_type", "(string) the type of the db.")
    .Arg(
        "keep_device",
        "(int, default 0) if nonzero, the blobs are loaded into the device that "
        "is specified in the serialized BlobProto. Otherwise, the device will be "
        "set as the one that the Load operator is being run under.")
    .Arg(
        "load_all",
        "(int, default 0) if nonzero, will load all blobs pointed to by the db "
        "to the workspace overwriting/creating blobs as needed.")
    .Arg(
        "allow_incomplete",
        "(bool, default false) if true, will allow not loading all the output "
        "blobs specified in the outputs")
    .Arg(
        "source_blob_names",
        "(list of strings) if set, used instead of output "
        "blob names, to specify which blobs in the db shall be loaded. Must be "
        "the same length as number of output blobs.");

OPERATOR_SCHEMA(Save)
    .NumInputs(1, INT_MAX)
    .NumOutputs(0)
    .SetDoc(R"DOC(
The Save operator saves a set of blobs to a db. It takes [1, infinity) number
of inputs and has no output. The contents of the inputs are written into the
db specified by the arguments.
)DOC")
    .Arg(
        "absolute_path",
        "(int, default 0) if set, use the db path directly and do not prepend "
        "the current root folder of the workspace.")
     .Arg(
         "strip_prefix",
         "(string, default=\"\") characters in the provided blob "
         " names that match strip_prefix will be removed prior to saving."
         " Also, characters that precede strip_prefix will be removed. Useful "
         " for removing device scope from blob names.")
    .Arg(
        "blob_name_overrides",
        "(list of strings) if set, used instead of original "
        "blob names. Must be the same length as number of blobs.")
    .Arg("db", "(string) the path to the db to load.")
    .Arg("db_type", "(string) the type of the db.");

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
