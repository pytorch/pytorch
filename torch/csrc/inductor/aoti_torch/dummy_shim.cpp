extern "C" {

// Common pointer helpers
void* aoti_torch_get_data_ptr(void*) { return nullptr; }
long* aoti_torch_get_sizes(void*) { return nullptr; }
long* aoti_torch_get_strides(void*) { return nullptr; }
long aoti_torch_get_storage_size(void*) { return 0; }
long aoti_torch_get_storage_offset(void*) { return 0; }
long aoti_torch_get_device_type(void*) { return 0; }

// Tensor creation / deletion
void* aoti_torch_new_tensor_handle() { return nullptr; }
void* aoti_torch_clone(void*) { return nullptr; }
void* aoti_torch_clone_preserve_strides(void*) { return nullptr; }
void* aoti_torch_empty_strided(int, const long*, const long*) { return nullptr; }
void aoti_torch_delete_tensor_object(void*) {}
void* aoti_torch_create_tensor_from_blob(void*, long*, long*, int) { return nullptr; }
void* aoti_torch_create_tensor_from_blob_v2(void*, long*, long*, int, long, void*) { return nullptr; }

// Stream guards
void* aoti_torch_create_cuda_stream_guard(void*) { return nullptr; }
void aoti_torch_delete_cuda_stream_guard(void*) {}

// Grad mode
bool aoti_torch_grad_mode_is_enabled() { return false; }
void aoti_torch_grad_mode_set_enabled(bool) {}

// Dtype / layout / device type constants
int aoti_torch_get_dtype(void*) { return 0; }
int aoti_torch_dtype_float32 = 0;
int aoti_torch_device_type_cpu = 0;
int aoti_torch_device_type_cuda = 1;
int aoti_torch_layout_strided = 0;

// Misc
void aoti_torch_warn(const char*) {}

} // extern "C"
