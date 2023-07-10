Here is the folder for APIs on rocm, which the backend code is from composable kernel.

Below is the introduction to the files.

"src/fmha.h" is the header file for the C++ APIs, in which declared the  function "run_fmha_fp16_bf16_gfx90a".

"fmha_api.cpp" is the c++ file that defined the API function "mha_fwd", this function will call function "run_fmha_fp16_bf16_gfx90a". This function also contains a main function to test with the API.

"src/fmha_fprop_fp16_bf16_kernel.gfx90a" is the interface that link API in fmha_api.cpp and the CK backend, which defined function "run_fmha_fp16_bf16_gfx90a". In this function, it will use parameters conveyed from "mha_fwd" to choose proper instance parameters for CK function. Function "run_fmha_fp16_bf16_gfx90a_loop_" will use parameters from "run_fmha_fp16_bf16_gfx90a" to initialize instance in CK and call CK function. 

"CMakeList.txt" is a cmake file to compile the example above.

Useage for "CMakeLists.txt": 
```
$mkdir build
$cd build
$cmake ..
$make
```

My docker is from https://hub.docker.com/layers/rocm/pytorch/rocm5.3.2_ubuntu20.04_py3.7_pytorch_1.12.1/images/sha256-387b2538d14cfd55a9510b7ea07049f1e71b7e755413080153b997c798fe5099?context=explore

If you choose another docker or you install pytorch by yourself.

Please change line 8 in CMakeLists.txt file with your own path.

You can use command
``` 
python -c 'import torch;print(torch.utils.cmake_prefix_path)'
```
to find your path.

Way to build with docker file:

Change the github username and tocken with that of yourself in line https://github.com/ROCmSoftwarePlatform/flash-attention_private/blob/41ddb2fb3884085ee5318d30f8e919944ee18745/csrc/flash_attn_rocm/Dockerfile#L11 firstly.

Then
```
sudo docker build -t flash_attention:rocm5.3.2 .
```

If you want to test the performance, you can set the parameter “time_kernel” as true. And then the kernel will run 10 times and give out the average running time. You can find the parameter in this line: https://github.com/ROCmSoftwarePlatform/flash-attention_private/blob/fb47a607682a873a3e0b17ae220849cc11a34d8b/csrc/flash_attn_rocm/src/fmha_fprop_fp16_bf16_kernel.gfx90a.cpp#L142

If you want to verify the results, you can set the parameter “do_verification” in this line https://github.com/ROCmSoftwarePlatform/flash-attention_private/blob/fb47a607682a873a3e0b17ae220849cc11a34d8b/csrc/flash_attn_rocm/fmha_api.cpp#L271 . And then the code can do the same computation on cpu and compare with the results from device and show whether device results are right.





