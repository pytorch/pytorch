#include <stdbool.h>
#include <stdint.h>

#include "NeuralNetworks.h"

#define DLNNAPI_FUNCTION_COUNT 23

#define DLNNAPI_FLAG_VERSION_MASK 0xFF
/* Android 8.1, API 27 version */
#define DLNNAPI_FLAG_VERSION_27 0x01

/* clang-format off */
/* nn api function types */
typedef int (*ANeuralNetworksMemory_createFromFd_fn)(size_t size, int protect, int fd, size_t offset, ANeuralNetworksMemory** memory);

typedef void (*ANeuralNetworksMemory_free_fn)(ANeuralNetworksMemory* memory);

typedef int (*ANeuralNetworksModel_create_fn)(ANeuralNetworksModel** model);

typedef int (*ANeuralNetworksModel_finish_fn)(ANeuralNetworksModel* model);

typedef void (*ANeuralNetworksModel_free_fn)(ANeuralNetworksModel* model);

typedef int (*ANeuralNetworksCompilation_create_fn)(ANeuralNetworksModel* model, ANeuralNetworksCompilation** compilation);

typedef void (*ANeuralNetworksCompilation_free_fn)(ANeuralNetworksCompilation* compilation);

typedef int (*ANeuralNetworksCompilation_setPreference_fn)(ANeuralNetworksCompilation* compilation, int32_t preference);

typedef int (*ANeuralNetworksCompilation_finish_fn)(ANeuralNetworksCompilation* compilation);

typedef int (*ANeuralNetworksModel_addOperand_fn)(ANeuralNetworksModel* model, const ANeuralNetworksOperandType* type);

typedef int (*ANeuralNetworksModel_setOperandValue_fn)(ANeuralNetworksModel* model, int32_t index, const void* buffer, size_t length);

typedef int (*ANeuralNetworksModel_setOperandValueFromMemory_fn)(ANeuralNetworksModel* model, int32_t index, const ANeuralNetworksMemory* memory, size_t offset, size_t length);

typedef int (*ANeuralNetworksModel_addOperation_fn)(ANeuralNetworksModel* model, ANeuralNetworksOperationType type, uint32_t inputCount, const uint32_t* inputs, uint32_t outputCount, const uint32_t* outputs);

typedef int (*ANeuralNetworksModel_identifyInputsAndOutputs_fn)(ANeuralNetworksModel* model, uint32_t inputCount, const uint32_t* inputs, uint32_t outputCount, const uint32_t* outputs);

typedef int (*ANeuralNetworksExecution_create_fn)(ANeuralNetworksCompilation* compilation, ANeuralNetworksExecution** execution);

typedef void (*ANeuralNetworksExecution_free_fn)(ANeuralNetworksExecution* execution);

typedef int (*ANeuralNetworksExecution_setInput_fn)(ANeuralNetworksExecution* execution, int32_t index, const ANeuralNetworksOperandType* type, const void* buffer, size_t length);

typedef int (*ANeuralNetworksExecution_setInputFromMemory_fn)(ANeuralNetworksExecution* execution, int32_t index, const ANeuralNetworksOperandType* type, const ANeuralNetworksMemory* memory, size_t offset, size_t length);

typedef int (*ANeuralNetworksExecution_setOutput_fn)(ANeuralNetworksExecution* execution, int32_t index, const ANeuralNetworksOperandType* type, void* buffer, size_t length);

typedef int (*ANeuralNetworksExecution_setOutputFromMemory_fn)(ANeuralNetworksExecution* execution, int32_t index, const ANeuralNetworksOperandType* type, const ANeuralNetworksMemory* memory, size_t offset, size_t length);

typedef int (*ANeuralNetworksExecution_startCompute_fn)(ANeuralNetworksExecution* execution, ANeuralNetworksEvent** event);

typedef int (*ANeuralNetworksEvent_wait_fn)(ANeuralNetworksEvent* event);

typedef void (*ANeuralNetworksEvent_free_fn)(ANeuralNetworksEvent* event);

struct dlnnapi {
	void* handle;
	uint32_t flags;
	union {
		struct {
			/* ndk-r16b */
			ANeuralNetworksMemory_createFromFd_fn             ANeuralNetworksMemory_createFromFd;
			ANeuralNetworksMemory_free_fn                     ANeuralNetworksMemory_free;
			ANeuralNetworksModel_create_fn                    ANeuralNetworksModel_create;
			ANeuralNetworksModel_finish_fn                    ANeuralNetworksModel_finish;
			ANeuralNetworksModel_free_fn                      ANeuralNetworksModel_free;
			ANeuralNetworksCompilation_create_fn              ANeuralNetworksCompilation_create;
			ANeuralNetworksCompilation_free_fn                ANeuralNetworksCompilation_free;
			ANeuralNetworksCompilation_setPreference_fn       ANeuralNetworksCompilation_setPreference;
			ANeuralNetworksCompilation_finish_fn              ANeuralNetworksCompilation_finish;
			ANeuralNetworksModel_addOperand_fn                ANeuralNetworksModel_addOperand;
			ANeuralNetworksModel_setOperandValue_fn           ANeuralNetworksModel_setOperandValue;
			ANeuralNetworksModel_setOperandValueFromMemory_fn ANeuralNetworksModel_setOperandValueFromMemory;
			ANeuralNetworksModel_addOperation_fn              ANeuralNetworksModel_addOperation;
			ANeuralNetworksModel_identifyInputsAndOutputs_fn  ANeuralNetworksModel_identifyInputsAndOutputs;
			ANeuralNetworksExecution_create_fn                ANeuralNetworksExecution_create;
			ANeuralNetworksExecution_free_fn                  ANeuralNetworksExecution_free;
			ANeuralNetworksExecution_setInput_fn              ANeuralNetworksExecution_setInput;
			ANeuralNetworksExecution_setInputFromMemory_fn    ANeuralNetworksExecution_setInputFromMemory;
			ANeuralNetworksExecution_setOutput_fn             ANeuralNetworksExecution_setOutput;
			ANeuralNetworksExecution_setOutputFromMemory_fn   ANeuralNetworksExecution_setOutputFromMemory;
			ANeuralNetworksExecution_startCompute_fn          ANeuralNetworksExecution_startCompute;
			ANeuralNetworksEvent_wait_fn                      ANeuralNetworksEvent_wait;
			ANeuralNetworksEvent_free_fn                      ANeuralNetworksEvent_free;
		};
		void* functions[DLNNAPI_FUNCTION_COUNT];
	};
};
/* clang-format on */

#ifdef __cplusplus
extern "C" {
#endif

bool dlnnapi_load(struct dlnnapi* nnapi, uint32_t flags);
void dlnnapi_free(struct dlnnapi* nnapi);

#ifdef __cplusplus
} /* extern "C" */
#endif
