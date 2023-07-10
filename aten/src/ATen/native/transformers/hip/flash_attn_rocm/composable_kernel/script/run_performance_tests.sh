#!/bin/bash 
#
# in order to run this script you'd first need to build the ckProfiler executable in ../build/bin/
# run the script as "./run_performance_tests.sh <verification> <tag for your test environment> <branch name> < node name>
# input arguments: 
# verification = 0 : do not verify result correctness on CPU
#              = 1 : verify correctness on CPU (may take a long time)
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
./profile_gemm.sh gemm 0 0 $verify 1 0 1 | tee -a $gemm_log
./profile_gemm.sh gemm 1 0 $verify 1 0 1 | tee -a $gemm_log
./profile_gemm.sh gemm 2 0 $verify 1 0 1 | tee -a $gemm_log
./profile_gemm.sh gemm 3 0 $verify 1 0 1 | tee -a $gemm_log
./profile_gemm.sh gemm 0 1 $verify 1 0 1 | tee -a $gemm_log
./profile_gemm.sh gemm 1 1 $verify 1 0 1 | tee -a $gemm_log
./profile_gemm.sh gemm 2 1 $verify 1 0 1 | tee -a $gemm_log
./profile_gemm.sh gemm 3 1 $verify 1 0 1 | tee -a $gemm_log
./profile_gemm.sh gemm 0 2 $verify 1 0 1 | tee -a $gemm_log
./profile_gemm.sh gemm 1 2 $verify 1 0 1 | tee -a $gemm_log
./profile_gemm.sh gemm 2 2 $verify 1 0 1 | tee -a $gemm_log
./profile_gemm.sh gemm 3 2 $verify 1 0 1 | tee -a $gemm_log
./profile_gemm.sh gemm 0 3 $verify 1 0 1 | tee -a $gemm_log
./profile_gemm.sh gemm 1 3 $verify 1 0 1 | tee -a $gemm_log
./profile_gemm.sh gemm 2 3 $verify 1 0 1 | tee -a $gemm_log
./profile_gemm.sh gemm 3 3 $verify 1 0 1 | tee -a $gemm_log

#run resnet50 tests
export resnet256_log="perf_resnet50_N256.log"
print_log_header $resnet256_log $env_type $branch $host_name
./profile_resnet50.sh conv_fwd_bias_relu 1 1 1 1 $verify 1 0 1 256 | tee -a $resnet256_log
export resnet4_log="perf_resnet50_N4.log"
print_log_header $resnet4_log $env_type $branch $host_name
./profile_resnet50.sh conv_fwd_bias_relu 1 1 1 1 $verify 1 0 1 4 | tee -a $resnet4_log
