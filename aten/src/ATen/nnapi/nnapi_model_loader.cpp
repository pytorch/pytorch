// NOLINTNEXTLINE(modernize-deprecated-headers)
#include <stdint.h>

#include <ATen/nnapi/NeuralNetworks.h>
#include <ATen/nnapi/nnapi_wrapper.h>
#include <ATen/nnapi/nnapi_model_loader.h>
#include <c10/util/irange.h>


#ifndef NNAPI_LOADER_STANDALONE

# include <c10/util/Logging.h>

#else

#define CAFFE_ENFORCE(cond, ...) do { if (!cond) { return -1; } } while (0)

#endif


#define NNAPI_CHECK(res) CAFFE_ENFORCE(res == ANEURALNETWORKS_NO_ERROR, "NNAPI returned error: ", res)


namespace caffe2 {
namespace nnapi {

namespace {

/*
Serialized format for NNAPI models.  It is basically just a list arguments
for calls to be made to NNAPI.
*/

typedef enum _SourceType {
  SOURCE_IMMEDIATE = 0,
  SOURCE_NUMBERED_BUFFER = 2,
  SOURCE_NUMBERED_MEMORY = 3,
} SourceType;

typedef struct _SerializedOperand {
  int32_t type;
  uint32_t dimension_count;
  float scale;
  int32_t zero_point;
} SerializedOperand;

typedef struct _SerializedValue {
  int32_t index;
  int32_t source_type;
  uint32_t source_length;
} SerializedValue;

typedef struct _SerializedOperation {
  int32_t operation_type;
  uint32_t input_count;
  uint32_t output_count;
} SerializedOperation;

typedef struct _SerializedModel {
  int32_t version;
  int32_t operand_count;
  int32_t value_count;
  int32_t operation_count;
  int32_t input_count;
  int32_t output_count;
  // SerializedOperand operands[operand_count];
  // SerializedValue values[value_count];
  // SerializedOperation operations[operation_count];
  // uint32_t operand_dimensions[sum(dimension_count)]
  // uint32_t value_data[sum(source_length+pad)/4]
  // uint32_t operation_args[sum(input_count + output_count)]
  // uint32_t model_inputs[input_count]
  // uint32_t model_outputs[output_count]
} SerializedModel;


/**
 * Get the physically stored size of a value.  All values are padded out
 * to a multiple of 4 bytes to ensure the next value is 4-byte aligned.
 */
static uint32_t value_physical_size(uint32_t len) {
  uint32_t phys = len;
  if (len % 4 == 0) {
    return len;
  }
  return len + 4 - (phys % 4);
}

} // namespace


int load_nnapi_model(
    struct nnapi_wrapper* nnapi,
    ANeuralNetworksModel* model,
    const void* serialized_model,
    int64_t model_length,
    size_t num_buffers,
    const void** buffer_ptrs,
    int32_t* buffer_sizes,
    size_t /*num_memories*/,
    ANeuralNetworksMemory** /*memories*/,
    int32_t* /*memory_sizes*/,
    int32_t* out_input_count,
    int32_t* out_output_count,
    size_t* out_bytes_consumed) {
  int64_t required_size = 0;
  const uint8_t* next_pointer = (const uint8_t*)serialized_model;
  const uint8_t* end_of_buf = (const uint8_t*)serialized_model + model_length;

  required_size += sizeof(SerializedModel);
  CAFFE_ENFORCE(model_length >= required_size, "Model is too small.  Size = ", model_length);
  const SerializedModel* ser_model = (SerializedModel*)next_pointer;
  next_pointer = (uint8_t*)serialized_model + required_size;
  CAFFE_ENFORCE(next_pointer <= end_of_buf);

  CAFFE_ENFORCE(ser_model->version == 1);
  // Keep these small to avoid integer overflow.
  CAFFE_ENFORCE(ser_model->operand_count    < (1 << 24));
  CAFFE_ENFORCE(ser_model->value_count      < (1 << 24));
  CAFFE_ENFORCE(ser_model->operation_count  < (1 << 24));
  CAFFE_ENFORCE(ser_model->input_count      < (1 << 24));
  CAFFE_ENFORCE(ser_model->output_count     < (1 << 24));

  required_size += sizeof(SerializedOperand) * ser_model->operand_count;
  CAFFE_ENFORCE(model_length >= required_size, "Model is too small.  Size = ", model_length);
  const SerializedOperand* operands = (const SerializedOperand*)next_pointer;
  next_pointer = (uint8_t*)serialized_model + required_size;
  CAFFE_ENFORCE(next_pointer <= end_of_buf);

  required_size += sizeof(SerializedValue) * ser_model->value_count;
  CAFFE_ENFORCE(model_length >= required_size, "Model is too small.  Size = ", model_length);
  const SerializedValue* values = (const SerializedValue*)next_pointer;
  next_pointer = (uint8_t*)serialized_model + required_size;
  CAFFE_ENFORCE(next_pointer <= end_of_buf);

  required_size += sizeof(SerializedOperation) * ser_model->operation_count;
  CAFFE_ENFORCE(model_length >= required_size, "Model is too small.  Size = ", model_length);
  const SerializedOperation* operations = (const SerializedOperation*)next_pointer;
  next_pointer = (uint8_t*)serialized_model + required_size;
  CAFFE_ENFORCE(next_pointer <= end_of_buf);

  for (const auto i : c10::irange(ser_model->operand_count)) {
    required_size += 4 * operands[i].dimension_count;
  }

  for (const auto i : c10::irange(ser_model->value_count)) {
    required_size += value_physical_size(values[i].source_length);
  }

  for (const auto i : c10::irange(ser_model->operation_count)) {
    required_size += 4 * (operations[i].input_count + operations[i].output_count);
  }

  required_size += 4 * (ser_model->input_count + ser_model->output_count);

  CAFFE_ENFORCE(model_length >= required_size, "Model is too small.  Size = ", model_length);
  CAFFE_ENFORCE(next_pointer <= end_of_buf);

  for (const auto i : c10::irange(ser_model->operand_count)) {
    ANeuralNetworksOperandType operand;
    operand.type = operands[i].type;
    operand.scale = operands[i].scale;
    operand.zeroPoint = operands[i].zero_point;
    operand.dimensionCount = operands[i].dimension_count;
    operand.dimensions = operands[i].dimension_count ? (const uint32_t*)next_pointer : nullptr;

    next_pointer += 4 * operands[i].dimension_count;
    CAFFE_ENFORCE(next_pointer <= end_of_buf);

    int result = nnapi->Model_addOperand(model, &operand);
    NNAPI_CHECK(result);
  }

  for (const auto i : c10::irange(ser_model->value_count)) {
    uint32_t len = values[i].source_length;
    const uint8_t* stored_pointer = next_pointer;
    const void* value_pointer = nullptr;
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    size_t value_length;

    switch ((SourceType)values[i].source_type) {
      case SOURCE_IMMEDIATE:
        {
          value_pointer = stored_pointer;
          value_length = len;
        }
        break;
      case SOURCE_NUMBERED_BUFFER:
        {
          CAFFE_ENFORCE(len == 12);
          uint32_t buffer_number = *(uint32_t*)stored_pointer;
          uint32_t buffer_offset = *(uint32_t*)(stored_pointer + 4);
          uint32_t operand_length = *(uint32_t*)(stored_pointer + 8);
          CAFFE_ENFORCE(buffer_number < num_buffers);
          CAFFE_ENFORCE(buffer_offset + operand_length >= buffer_offset);  // No integer overflow
          CAFFE_ENFORCE(buffer_offset + operand_length <= (uint32_t)buffer_sizes[buffer_number]);  // No buffer overflow
          value_pointer = (uint8_t*)buffer_ptrs[buffer_number] + buffer_offset;
          value_length = operand_length;
        }
        break;
      case SOURCE_NUMBERED_MEMORY:
        CAFFE_ENFORCE(false, "Memory inputs not implemented yet.");
        break;
      default:
        CAFFE_ENFORCE(false, "Unknown source type: ", values[i].source_type);
    }

    CAFFE_ENFORCE(value_pointer != nullptr);

    next_pointer += value_physical_size(len);
    CAFFE_ENFORCE(next_pointer <= end_of_buf);

    int result = nnapi->Model_setOperandValue(
        model,
        values[i].index,
        value_pointer,
        value_length);
    NNAPI_CHECK(result);
  }

  for (const auto i : c10::irange(ser_model->operation_count)) {
    const uint32_t* inputs = (const uint32_t*)next_pointer;
    next_pointer += 4 * operations[i].input_count;
    CAFFE_ENFORCE(next_pointer <= end_of_buf);
    const uint32_t* outputs = (const uint32_t*)next_pointer;
    next_pointer += 4 * operations[i].output_count;
    CAFFE_ENFORCE(next_pointer <= end_of_buf);

    int result = nnapi->Model_addOperation(
        model,
        operations[i].operation_type,
        operations[i].input_count,
        inputs,
        operations[i].output_count,
        outputs);
    NNAPI_CHECK(result);
  }

  const uint32_t* model_inputs = (const uint32_t*)next_pointer;
  next_pointer += 4 * ser_model->input_count;
  CAFFE_ENFORCE(next_pointer <= end_of_buf);
  const uint32_t* model_outputs = (const uint32_t*)next_pointer;
  next_pointer += 4 * ser_model->output_count;
  CAFFE_ENFORCE(next_pointer <= end_of_buf);

  int result = nnapi->Model_identifyInputsAndOutputs(
      model,
      ser_model->input_count,
      model_inputs,
      ser_model->output_count,
      model_outputs);
  NNAPI_CHECK(result);

  *out_input_count = ser_model->input_count;
  *out_output_count = ser_model->output_count;

  // TODO: Maybe eliminate required_size and just rely on next_pointer for bounds checking.
  CAFFE_ENFORCE(next_pointer <= end_of_buf);
  CAFFE_ENFORCE(next_pointer == (const uint8_t*)serialized_model + required_size);
  if (out_bytes_consumed != nullptr) {
    *out_bytes_consumed = next_pointer - (const uint8_t*)serialized_model;
  }

  return 0;
}

}} // namespace caffe2::nnapi
