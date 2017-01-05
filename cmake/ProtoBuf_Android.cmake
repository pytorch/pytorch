# This cmake file manually builds the protobuf library from the third_party
# subfolder, which is useful in the cases of cross compilation. Note that
# when we are cross compling, the user is responsible for providing a protoc
# binary that matches the version of the third_party/protobuf library.

message(WARNING "protobuf build with local third_party library: wip")
if (NOT PROTOBUF_PROTOC_EXECUTABLE)
  message(FATAL_ERROR "For Android build, you will need to manually provide PROTOBUF_PROTOC_EXECUTABLE")
endif()
include_directories(SYSTEM "${PROJECT_SOURCE_DIR}/third_party/protobuf/src") 

file(GLOB_RECURSE Caffe2_THIRD_PARTY_PROTOBUF_HDRS "${PROJECT_SOURCE_DIR}/third_party/protobuf/src/google/protobuf/*.h")

# TODO: allow the use of lite proto.
set(Caffe2_THIRD_PARTY_PROTOBUF_SRCS
    # Lite srcs
	"protobuf/src/google/protobuf/arena.cc"
	"protobuf/src/google/protobuf/arenastring.cc"
	"protobuf/src/google/protobuf/extension_set.cc"
	"protobuf/src/google/protobuf/generated_message_util.cc"
	"protobuf/src/google/protobuf/io/coded_stream.cc"
	"protobuf/src/google/protobuf/io/zero_copy_stream.cc"
	"protobuf/src/google/protobuf/io/zero_copy_stream_impl_lite.cc"
	"protobuf/src/google/protobuf/message_lite.cc"
	"protobuf/src/google/protobuf/repeated_field.cc"
	"protobuf/src/google/protobuf/stubs/atomicops_internals_x86_gcc.cc"
	"protobuf/src/google/protobuf/stubs/atomicops_internals_x86_msvc.cc"
	"protobuf/src/google/protobuf/stubs/bytestream.cc"
	"protobuf/src/google/protobuf/stubs/common.cc"
	"protobuf/src/google/protobuf/stubs/int128.cc"
	"protobuf/src/google/protobuf/stubs/once.cc"
	"protobuf/src/google/protobuf/stubs/status.cc"
	"protobuf/src/google/protobuf/stubs/statusor.cc"
	"protobuf/src/google/protobuf/stubs/stringpiece.cc"
	"protobuf/src/google/protobuf/stubs/stringprintf.cc"
	"protobuf/src/google/protobuf/stubs/structurally_valid.cc"
	"protobuf/src/google/protobuf/stubs/strutil.cc"
	"protobuf/src/google/protobuf/stubs/time.cc"
	"protobuf/src/google/protobuf/wire_format_lite.cc"
    # full srcs
	"protobuf/src/google/protobuf/any.cc"
	"protobuf/src/google/protobuf/any.pb.cc"
	"protobuf/src/google/protobuf/api.pb.cc"
	"protobuf/src/google/protobuf/compiler/importer.cc"
	"protobuf/src/google/protobuf/compiler/parser.cc"
	"protobuf/src/google/protobuf/descriptor.cc"
	"protobuf/src/google/protobuf/descriptor.pb.cc"
	"protobuf/src/google/protobuf/descriptor_database.cc"
	"protobuf/src/google/protobuf/duration.pb.cc"
	"protobuf/src/google/protobuf/dynamic_message.cc"
	"protobuf/src/google/protobuf/empty.pb.cc"
	"protobuf/src/google/protobuf/extension_set_heavy.cc"
	"protobuf/src/google/protobuf/field_mask.pb.cc"
	"protobuf/src/google/protobuf/generated_message_reflection.cc"
	"protobuf/src/google/protobuf/io/gzip_stream.cc"
	"protobuf/src/google/protobuf/io/printer.cc"
	"protobuf/src/google/protobuf/io/strtod.cc"
	"protobuf/src/google/protobuf/io/tokenizer.cc"
	"protobuf/src/google/protobuf/io/zero_copy_stream_impl.cc"
	"protobuf/src/google/protobuf/map_field.cc"
	"protobuf/src/google/protobuf/message.cc"
	"protobuf/src/google/protobuf/reflection_ops.cc"
	"protobuf/src/google/protobuf/service.cc"
	"protobuf/src/google/protobuf/source_context.pb.cc"
	"protobuf/src/google/protobuf/struct.pb.cc"
	"protobuf/src/google/protobuf/stubs/mathlimits.cc"
	"protobuf/src/google/protobuf/stubs/substitute.cc"
	"protobuf/src/google/protobuf/text_format.cc"
	"protobuf/src/google/protobuf/timestamp.pb.cc"
	"protobuf/src/google/protobuf/type.pb.cc"
	"protobuf/src/google/protobuf/unknown_field_set.cc"
	"protobuf/src/google/protobuf/util/field_comparator.cc"
	"protobuf/src/google/protobuf/util/field_mask_util.cc"
	"protobuf/src/google/protobuf/util/internal/datapiece.cc"
	"protobuf/src/google/protobuf/util/internal/default_value_objectwriter.cc"
	"protobuf/src/google/protobuf/util/internal/error_listener.cc"
	"protobuf/src/google/protobuf/util/internal/field_mask_utility.cc"
	"protobuf/src/google/protobuf/util/internal/json_escaping.cc"
	"protobuf/src/google/protobuf/util/internal/json_objectwriter.cc"
	"protobuf/src/google/protobuf/util/internal/json_stream_parser.cc"
	"protobuf/src/google/protobuf/util/internal/object_writer.cc"
	"protobuf/src/google/protobuf/util/internal/protostream_objectsource.cc"
	"protobuf/src/google/protobuf/util/internal/protostream_objectwriter.cc"
	"protobuf/src/google/protobuf/util/internal/type_info.cc"
	"protobuf/src/google/protobuf/util/internal/type_info_test_helper.cc"
	"protobuf/src/google/protobuf/util/internal/utility.cc"
	"protobuf/src/google/protobuf/util/json_util.cc"
	"protobuf/src/google/protobuf/util/message_differencer.cc"
	"protobuf/src/google/protobuf/util/time_util.cc"
	"protobuf/src/google/protobuf/util/type_resolver_util.cc"
	"protobuf/src/google/protobuf/wire_format.cc"
	"protobuf/src/google/protobuf/wrappers.pb.cc"
)

prepend(Caffe2_THIRD_PARTY_PROTOBUF_SRCS "${PROJECT_SOURCE_DIR}/third_party/" "${Caffe2_THIRD_PARTY_PROTOBUF_SRCS}")

# TODO: having HAVE_PTHREAD in the whole CXX_FLAGS might be suboptimal
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DHAVE_PTHREAD")
add_library(Caffe2_THIRD_PARTY_PROTOBUF STATIC ${Caffe2_THIRD_PARTY_PROTOBUF_HDRS} ${Caffe2_THIRD_PARTY_PROTOBUF_SRCS})
target_include_directories(Caffe2_THIRD_PARTY_PROTOBUF PUBLIC ${PROJECT_SOURCE_DIR}/third_party/protobuf/src)
list(APPEND Caffe2_MAIN_LIBS Caffe2_THIRD_PARTY_PROTOBUF)
