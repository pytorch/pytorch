#!/bin/bash 
#
# in order to run this script you'd first need to build the ckProfiler executable in ../build/bin/
# you would also need to set up some environment variables in order to 
# post your new test results to the database and compare them to the baseline
# please contact Illia.Silin@amd.com for more details
#
# run the script as "./run_full_performance_tests.sh <verification> <tag for your test environment> <branch name> < node name>
# input arguments: 
# verification = 0 : do not verify result correctness on CPU
#              = 1 : verifuy correctness on CPU (may take a long time)
# environment tag  : a string describing the specifics of your test environment
# branch name      : name of the branch in git repo (git status | grep -e 'On branch')
# node name        : $hostname

#get the command line arguments:
export verify=$1
echo 'Verification: ' $verify
export env_type=$2
echo 'Environment type: ' $env_type
export branch=$3
echo 'Branch name: ' $branch
export host_name=$4
echo 'Host name: ' $host_name
function print_log_header(){
	rm -f $1;
	echo 'On branch ' $3 &> $1;
	echo 'Node name: ' $4 >> $1;
	#get GPU_arch and number of compute units from rocminfo
	echo -n "GPU_arch: " >> $1; rocminfo | grep "Name:" | grep "gfx" >> $1;
	rocminfo | grep "Compute Unit:" >> $1;
	hipcc --version | grep -e 'HIP version'  >> $1;
	echo 'Environment type: ' $2 >> $1;
	/opt/rocm/bin/amdclang++ --version | grep -e 'InstalledDir' >> $1;
}

#run gemm tests
export gemm_log="perf_gemm.log"
print_log_header $gemm_log $env_type $branch $host_name
./profile_gemm.sh gemm 0 0 $verify 1 0 1 2>&1 | tee -a $gemm_log
./profile_gemm.sh gemm 1 0 $verify 1 0 1 2>&1 | tee -a $gemm_log
./profile_gemm.sh gemm 2 0 $verify 1 0 1 2>&1 | tee -a $gemm_log
./profile_gemm.sh gemm 3 0 $verify 1 0 1 2>&1 | tee -a $gemm_log
./profile_gemm.sh gemm 0 1 $verify 1 0 1 2>&1 | tee -a $gemm_log
./profile_gemm.sh gemm 1 1 $verify 1 0 1 2>&1 | tee -a $gemm_log
./profile_gemm.sh gemm 2 1 $verify 1 0 1 2>&1 | tee -a $gemm_log
./profile_gemm.sh gemm 3 1 $verify 1 0 1 2>&1 | tee -a $gemm_log
./profile_gemm.sh gemm 0 2 $verify 1 0 1 2>&1 | tee -a $gemm_log
./profile_gemm.sh gemm 1 2 $verify 1 0 1 2>&1 | tee -a $gemm_log
./profile_gemm.sh gemm 2 2 $verify 1 0 1 2>&1 | tee -a $gemm_log
./profile_gemm.sh gemm 3 2 $verify 1 0 1 2>&1 | tee -a $gemm_log
./profile_gemm.sh gemm 0 3 $verify 1 0 1 2>&1 | tee -a $gemm_log
./profile_gemm.sh gemm 1 3 $verify 1 0 1 2>&1 | tee -a $gemm_log
./profile_gemm.sh gemm 2 3 $verify 1 0 1 2>&1 | tee -a $gemm_log
./profile_gemm.sh gemm 3 3 $verify 1 0 1 2>&1 | tee -a $gemm_log

#run batched_gemm tests
export batched_gemm_log="perf_batched_gemm.log"
print_log_header $batched_gemm_log $env_type $branch $host_name
./profile_batched_gemm.sh batched_gemm 0 0 $verify 1 0 1 2>&1 | tee -a $batched_gemm_log
./profile_batched_gemm.sh batched_gemm 0 1 $verify 1 0 1 2>&1 | tee -a $batched_gemm_log
./profile_batched_gemm.sh batched_gemm 0 2 $verify 1 0 1 2>&1 | tee -a $batched_gemm_log
./profile_batched_gemm.sh batched_gemm 0 3 $verify 1 0 1 2>&1 | tee -a $batched_gemm_log
./profile_batched_gemm.sh batched_gemm 1 0 $verify 1 0 1 2>&1 | tee -a $batched_gemm_log
./profile_batched_gemm.sh batched_gemm 1 1 $verify 1 0 1 2>&1 | tee -a $batched_gemm_log
./profile_batched_gemm.sh batched_gemm 1 2 $verify 1 0 1 2>&1 | tee -a $batched_gemm_log
./profile_batched_gemm.sh batched_gemm 1 3 $verify 1 0 1 2>&1 | tee -a $batched_gemm_log
./profile_batched_gemm.sh batched_gemm 2 0 $verify 1 0 1 2>&1 | tee -a $batched_gemm_log
./profile_batched_gemm.sh batched_gemm 2 1 $verify 1 0 1 2>&1 | tee -a $batched_gemm_log
./profile_batched_gemm.sh batched_gemm 2 2 $verify 1 0 1 2>&1 | tee -a $batched_gemm_log
./profile_batched_gemm.sh batched_gemm 2 3 $verify 1 0 1 2>&1 | tee -a $batched_gemm_log
./profile_batched_gemm.sh batched_gemm 3 0 $verify 1 0 1 2>&1 | tee -a $batched_gemm_log
./profile_batched_gemm.sh batched_gemm 3 1 $verify 1 0 1 2>&1 | tee -a $batched_gemm_log
./profile_batched_gemm.sh batched_gemm 3 2 $verify 1 0 1 2>&1 | tee -a $batched_gemm_log
./profile_batched_gemm.sh batched_gemm 3 3 $verify 1 0 1 2>&1 | tee -a $batched_gemm_log

#run grouped_gemm tests
export grouped_gemm_log="perf_grouped_gemm.log"
print_log_header $grouped_gemm_log $env_type $branch $host_name
./profile_grouped_gemm.sh grouped_gemm 1 0 $verify 1 0 1 2>&1 | tee -a $grouped_gemm_log
./profile_grouped_gemm.sh grouped_gemm 1 1 $verify 1 0 1 2>&1 | tee -a $grouped_gemm_log
./profile_grouped_gemm.sh grouped_gemm 1 2 $verify 1 0 1 2>&1 | tee -a $grouped_gemm_log
./profile_grouped_gemm.sh grouped_gemm 1 3 $verify 1 0 1 2>&1 | tee -a $grouped_gemm_log

#run GEMM+Bilinear tests
export gemm_bilinear_log="perf_gemm_bilinear.log"
print_log_header $gemm_bilinear_log $env_type $branch $host_name
./profile_gemm_bilinear.sh gemm_bilinear 1 0 $verify 1 0 1 2>&1 | tee -a $gemm_bilinear_log
./profile_gemm_bilinear.sh gemm_bilinear 1 1 $verify 1 0 1 2>&1 | tee -a $gemm_bilinear_log
./profile_gemm_bilinear.sh gemm_bilinear 1 2 $verify 1 0 1 2>&1 | tee -a $gemm_bilinear_log
./profile_gemm_bilinear.sh gemm_bilinear 1 3 $verify 1 0 1 2>&1 | tee -a $gemm_bilinear_log

#run conv_fwd tests
export conv_fwd_log="perf_conv_fwd.log"
print_log_header $conv_fwd_log $env_type $branch $host_name
./profile_conv_fwd.sh conv_fwd 0 1 $verify 1 0 1 256 2>&1 | tee -a $conv_fwd_log
./profile_conv_fwd.sh conv_fwd 1 1 $verify 1 0 1 256 2>&1 | tee -a $conv_fwd_log
./profile_conv_fwd.sh conv_fwd 2 1 $verify 1 0 1 256 2>&1 | tee -a $conv_fwd_log
./profile_conv_fwd.sh conv_fwd 3 1 $verify 1 0 1 256 2>&1 | tee -a $conv_fwd_log

#run conv_bwd_data tests
export conv_bwd_data_log="perf_conv_bwd_data.log"
print_log_header $conv_bwd_data_log $env_type $branch $host_name
./profile_conv_bwd_data.sh conv_bwd_data 0 1 $verify 1 0 1 256 2>&1 | tee -a $conv_bwd_data_log
./profile_conv_bwd_data.sh conv_bwd_data 1 1 $verify 1 0 1 256 2>&1 | tee -a $conv_bwd_data_log
./profile_conv_bwd_data.sh conv_bwd_data 2 1 $verify 1 0 1 256 2>&1 | tee -a $conv_bwd_data_log
./profile_conv_bwd_data.sh conv_bwd_data 3 1 $verify 1 0 1 256 2>&1 | tee -a $conv_bwd_data_log

#run resnet50 tests
export resnet256_log="perf_resnet50_N256.log"
print_log_header $resnet256_log $env_type $branch $host_name
./profile_resnet50.sh conv_fwd_bias_relu 1 1 1 1 $verify 1 0 1 256 2>&1 | tee -a $resnet256_log
export resnet4_log="perf_resnet50_N4.log"
print_log_header $resnet4_log $env_type $branch $host_name
./profile_resnet50.sh conv_fwd_bias_relu 1 1 1 1 $verify 1 0 1 4 2>&1 | tee -a $resnet4_log

#run reduction tests
export reduction_log="perf_reduction.log"
print_log_header $reduction_log $env_type $branch $host_name
./profile_reduce_with_index.sh $verify 2 10 --half 2>&1 | tee -a $reduction_log
./profile_reduce_no_index.sh $verify 2 10 --half 2>&1 | tee -a $reduction_log

#run splitK_gemm tests, first correctness verification, then performance
export splitK_gemm_ver_log="perf_splitK_gemm_verify.log"
print_log_header $splitK_gemm_ver_log $env_type $branch $host_name
./profile_splitK_gemm.sh gemm_splitk 0 0 $verify 1 0 0 4 2>&1 | tee -a $splitK_gemm_ver_log
./profile_splitK_gemm.sh gemm_splitk 0 1 $verify 1 0 0 4 2>&1 | tee -a $splitK_gemm_ver_log
./profile_splitK_gemm.sh gemm_splitk 0 2 $verify 1 0 0 4 2>&1 | tee -a $splitK_gemm_ver_log
./profile_splitK_gemm.sh gemm_splitk 0 3 $verify 1 0 0 4 2>&1 | tee -a $splitK_gemm_ver_log
./profile_splitK_gemm.sh gemm_splitk 1 0 $verify 1 0 0 4 2>&1 | tee -a $splitK_gemm_ver_log
./profile_splitK_gemm.sh gemm_splitk 1 1 $verify 1 0 0 4 2>&1 | tee -a $splitK_gemm_ver_log
./profile_splitK_gemm.sh gemm_splitk 1 2 $verify 1 0 0 4 2>&1 | tee -a $splitK_gemm_ver_log
./profile_splitK_gemm.sh gemm_splitk 1 3 $verify 1 0 0 4 2>&1 | tee -a $splitK_gemm_ver_log
export splitK_gemm_log="perf_splitK_gemm.log"
print_log_header $splitK_gemm_log $env_type $branch $host_name
./profile_splitK_gemm.sh gemm_splitk 0 0 0 1 0 1 4 2>&1 | tee -a $splitK_gemm_log
./profile_splitK_gemm.sh gemm_splitk 0 1 0 1 0 1 4 2>&1 | tee -a $splitK_gemm_log
./profile_splitK_gemm.sh gemm_splitk 0 2 0 1 0 1 4 2>&1 | tee -a $splitK_gemm_log
./profile_splitK_gemm.sh gemm_splitk 0 3 0 1 0 1 4 2>&1 | tee -a $splitK_gemm_log
./profile_splitK_gemm.sh gemm_splitk 1 0 0 1 0 1 4 2>&1 | tee -a $splitK_gemm_log
./profile_splitK_gemm.sh gemm_splitk 1 1 0 1 0 1 4 2>&1 | tee -a $splitK_gemm_log
./profile_splitK_gemm.sh gemm_splitk 1 2 0 1 0 1 4 2>&1 | tee -a $splitK_gemm_log
./profile_splitK_gemm.sh gemm_splitk 1 3 0 1 0 1 4 2>&1 | tee -a $splitK_gemm_log

#run ONNX gemm tests
export onnx_log="perf_onnx_gemm.log"
print_log_header $onnx_log $env_type $branch $host_name
./profile_onnx_gemm.sh gemm 0 0 $verify 1 0 1 2>&1 | tee -a $onnx_log
./profile_onnx_gemm.sh gemm 1 0 $verify 1 0 1 2>&1 | tee -a $onnx_log
