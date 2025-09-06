// Windows for #include <dlfcn.h>
#include <windows.h>
#include <stdio.h>

#include <iostream>
#include <memory>
#include <vector>
#include <string>

// Include the AOTInductor headers
// #include <torch/csrc/inductor/aoti_runner/model_container_runner_cuda.h>
#include <torch/csrc/inductor/aoti_runtime/interface.h>
// #include <torch/csrc/inductor/aoti_runtime/model_container.h>
// #include <torch/csrc/inductor/aoti_torch/tensor_converter.h> // @manual
#include <torch/csrc/inductor/aoti_torch/c/shim.h>
#include <standalone/slim/core/Empty.h>
#include <standalone/slim/cuda/Guard.h>
#include <standalone/torch/csrc/inductor/aoti_torch/tensor_converter.h>

static std::wstring u8u16(const char* s) {
    int len = MultiByteToWideChar(CP_UTF8, 0, s, -1, NULL, 0);
    std::wstring wbuf(len, L'\0');
    MultiByteToWideChar(CP_UTF8, 0, s, -1, &wbuf[0], len);
    if (!wbuf.empty() && wbuf.back() == L'\0') {
        wbuf.pop_back();
    }
    return wbuf;
}

int main() {
  try {

    // Load the DLL (model.pyd is a DLL on Windows)
HMODULE handle = nullptr;
{
    auto wname = u8u16(R"(C:\Users\shangdiy\source\repos\pytorch\model2\model.pyd)");

    // Try LoadLibraryExW with safe search flags if supported
    if (GetProcAddress(GetModuleHandleW(L"KERNEL32.DLL"), "AddDllDirectory") != NULL) {
        handle = LoadLibraryExW(
            wname.c_str(),
            NULL,
            LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
    }

    // Fallback if that failed
    if (!handle) {
        handle = LoadLibraryW(wname.c_str());
    }

    if (!handle) {
        DWORD dw = GetLastError();
        char buf[512];
        FormatMessageA(FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
                       NULL, dw, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
                       buf, sizeof(buf), NULL);
        std::cerr << "Failed to load model.pyd. WinError " << dw << ": " << buf << std::endl;
        return 1;
    } else {
        std::cout << "Loaded model.pyd" << std::endl;
    }
}
    decltype(&AOTInductorModelContainerCreateWithDevice) create_model{nullptr}; 
    decltype(&AOTInductorModelContainerDelete) delete_model{nullptr}; 
    decltype(&AOTInductorModelContainerRun) run_model{nullptr};


#define AOTI_LOAD_SYMBOL(handle_, var, name_str) \
    var = reinterpret_cast<decltype(var)>(GetProcAddress(handle_, name_str)); \
    if (!var) { \
        throw std::runtime_error("Could not GetProcAddress " name_str); \
    }

        AOTI_LOAD_SYMBOL(handle, create_model, "AOTInductorModelContainerCreateWithDevice");
        AOTI_LOAD_SYMBOL(handle, run_model, "AOTInductorModelContainerRun");
        AOTI_LOAD_SYMBOL(handle, delete_model, "AOTInductorModelContainerDelete");
#undef AOTI_LOAD_SYMBOL

    // Create array of input/output handles
        slim::SlimTensor x = slim::empty({8, 10}, c10::kFloat, c10::Device(c10::kCUDA, 0));
        float fill_value = 1.0;
        x.fill_(fill_value);
    // AOTInductorModel::run will steal the ownership of the input and output
    // tensor pointers
        std::vector<slim::SlimTensor> inputs = {x};
        std::vector<AtenTensorHandle> input_handles =
            unsafe_alloc_new_handles_from_tensors(inputs);

        AtenTensorHandle output_handle;
        AOTInductorModelContainerHandle container_handle;
        cudaStream_t stream = slim::cuda::getCurrentCUDAStream(0);
        // aoti_torch_get_current_cuda_stream(0, (void**)&stream);

        // Reinterpret as the opaque handle for AOTInductor
        AOTInductorStreamHandle stream_handle = reinterpret_cast<AOTInductorStreamHandle>(stream);

        // Construct model
       const char* cubin_dir = R"(C:\Users\shangdiy\source\repos\pytorch\model2\)";
        AOTIRuntimeError err =
            create_model(&container_handle, 1, "cuda", cubin_dir);
        if (err != AOTI_RUNTIME_SUCCESS) {
          throw std::runtime_error("Failed to create model container");
        } else {
          std::cout << "Created model\n";
        }

        // Run the model
        err = run_model(container_handle, input_handles.data(),
                          1, // num_inputs
                          &output_handle,
                          1,       // num_outputs
                          stream_handle, // stream
                          nullptr  // proxy_executor
        );
        if (err != AOTI_RUNTIME_SUCCESS) {
          throw std::runtime_error("Failed to run model");
        } else {
          std::cout << "Finish model\n";
        }

        std::vector<slim::SlimTensor> outputs =
            alloc_tensors_by_stealing_from_handles(&output_handle, 1);

    // Print the result
    slim::SlimTensor slim_tensor = outputs[0];
    auto slim_cpu = slim_tensor.cpu();
    float *slim_data = static_cast<float *>(slim_cpu.data_ptr());
     std::cout << "Output" << std::endl;
     std::cout << "slim_data ptr: " << slim_data << "\n";
    size_t num_elements = slim_cpu.numel(); // or equivalent method
     std::cout << num_elements << std::endl;

    for (size_t i = 0; i <  num_elements; ++i) {
      std::cout << slim_data[i] << "\n";
    }

    std::cout << "Done" << std::endl;

    delete_model(container_handle);
    FreeLibrary(handle);

    return 0;
  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }
}