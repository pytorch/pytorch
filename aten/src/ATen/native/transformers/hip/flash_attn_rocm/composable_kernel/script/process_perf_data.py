#!/usr/bin/env python3
import os, io, argparse, datetime
#import numpy as np
import sqlalchemy
from sqlalchemy.types import NVARCHAR, Float, Integer
import pymysql
import pandas as pd
from sshtunnel import SSHTunnelForwarder

def print_to_string(*args, **kwargs):
    output = io.StringIO()
    print(*args, file=output, **kwargs)
    contents = output.getvalue()
    output.close()
    return contents

def parse_args():
    parser = argparse.ArgumentParser(description='Parse results from tf benchmark runs')
    parser.add_argument('filename', type=str, help='Log file to prase or directory containing log files')
    args = parser.parse_args()
    files = []
    if os.path.isdir(args.filename):
        all_files = os.listdir(args.filename)
        for name in all_files:
            if not 'log' in name:
                continue
            files.append(os.path.join(args.filename, name))
    else:
        files = [args.filename]
    args.files = files
    return args

def get_log_params(logfile):
    print("logfile=",logfile)
    branch_name=' '
    node_id=' '
    gpu_arch=' '
    hip_vers=' '
    compute_units=0
    environment=' '
    rocm_vers=' '
    for line in open(logfile):
        if 'Branch name' in line:
            lst=line.split()
            branch_name=lst[2]
        if 'On branch' in line:
            lst=line.split()
            branch_name=lst[2]
        if 'Node name' in line:
            lst=line.split()
            node_id=lst[2]
        if 'GPU_arch' in line:
            lst=line.split()
            gpu_arch=lst[2]
        if 'HIP version' in line:
            lst=line.split()
            hip_vers=lst[2]
        if 'Compute Unit' in line:
            lst=line.split()
            compute_units=lst[2]
        if 'Environment type' in line:
            lst=line.split()
            environment=lst[2]
        if 'InstalledDir' in line:
            lst=line.split()
            rocm_vers=lst[1][lst[1].find('/opt/rocm-')+len('/opt/rocm-'):lst[1].rfind('/llvm/bin')]
    return branch_name, node_id, gpu_arch, compute_units, rocm_vers, hip_vers, environment

def parse_logfile(logfile):
    glue=''
    res=[]
    tests=[]
    kernels=[]
    tflops=[]
    dtype=[]
    alayout=[]
    blayout=[]
    M=[]
    N=[]
    K=[]
    StrideA=[]
    StrideB=[]
    StrideC=[]
    if 'perf_gemm.log' in logfile:
        for line in open(logfile):
            if 'Best Perf' in line:
                lst=line.split()
                if len(lst)>=37: #the line is complete
                    tests.append(glue.join(lst[5:30]))
                    kernels.append(glue.join(lst[37:]))
                    tflops.append(lst[33])
                    dtype.append(lst[5])
                    alayout.append(lst[8])
                    blayout.append(lst[11])
                    M.append(lst[14])
                    N.append(lst[17])
                    K.append(lst[20])
                    StrideA.append(lst[23])
                    StrideB.append(lst[26])
                    StrideC.append(lst[29])
                elif len(lst)<37 and len(lst)>=33: #the tflops are available
                    tests.append(glue.join(lst[5:30]))
                    kernels.append("N/A")
                    tflops.append(lst[33])
                    dtype.append(lst[5])
                    alayout.append(lst[8])
                    blayout.append(lst[11])
                    M.append(lst[14])
                    N.append(lst[17])
                    K.append(lst[20])
                    StrideA.append(lst[23])
                    StrideB.append(lst[26])
                    StrideC.append(lst[29])
                    print("warning: incomplete line:",lst)
                elif len(lst)<33: #even the tflops are not available
                    print("Error in ckProfiler output!")
                    print("warning: incomplete line=",lst)
        #sort results
        #sorted_tests = sorted(tests)
        res = [x for _,x in sorted(zip(tests,tflops))]
        #sorted_kernels = [x for _,x in sorted(zip(tests,kernels))]
        test_list=list(range(1,len(tests)+1))
    #parse conv_fwd and conv_bwd performance tests:
    elif 'conv_fwd' in logfile or 'conv_bwd_data' in logfile:
        for line in open(logfile):
            if 'tflops:' in line:
                lst=line.split()
                res.append(lst[1])
    #parse all other performance tests:
    elif 'resnet50' in logfile or 'batched_gemm' in logfile or 'grouped_gemm' in logfile  or 'gemm_bilinear' in logfile or 'reduction' in logfile:
        for line in open(logfile):
            if 'Best Perf' in line:
                lst=line.split()
                res.append(lst[4])
    elif 'onnx_gemm' in logfile or 'splitK_gemm' in logfile:
        for line in open(logfile):
            if 'Best Perf' in line:
                lst=line.split()
                res.append(lst[33])
    return res


def get_baseline(table, connection):
    query = '''SELECT * from '''+table+''' WHERE Datetime = (SELECT MAX(Datetime) FROM '''+table+''' where Branch_ID='develop' );'''
    return pd.read_sql_query(query, connection)

def store_new_test_result(table_name, test_results, testlist, branch_name, node_id, gpu_arch, compute_units, rocm_vers, hip_vers, environment, connection):
    params=[str(branch_name),str(node_id),str(gpu_arch),compute_units,str(rocm_vers),str(hip_vers),str(environment),str(datetime.datetime.now())]
    df=pd.DataFrame(data=[params],columns=['Branch_ID','Node_ID','GPU_arch','Compute Units','ROCM_version','HIP_version','Environment','Datetime'])
    df_add=pd.DataFrame(data=[test_results],columns=testlist)
    df=pd.concat([df,df_add],axis=1)
    #print("new test results dataframe:",df)
    df.to_sql(table_name,connection,if_exists='append',index=False)
    return 0

def compare_test_to_baseline(baseline,test,testlist):
    regression=0
    if not baseline.empty:
        base=baseline[testlist].to_numpy(dtype='float')
        base_list=base[0]
        ave_perf=0
        for i in range(len(base_list)):
            # success criterion:
            if base_list[i]>1.01*float(test[i]):
                print("test # ",i,"shows regression by {:.3f}%".format(
                    (float(test[i])-base_list[i])/base_list[i]*100))
                regression=1
            if base_list[i]>0: ave_perf=ave_perf+float(test[i])/base_list[i]
        if regression==0:
            print("no regressions found")
        ave_perf=ave_perf/len(base_list)
        print("average performance relative to baseline:",ave_perf)
    else:
        print("could not find a baseline")
    return regression

'''
def post_test_params(tlist,connection):
    sorted_dtypes = [x for _,x in sorted(zip(tests,dtype))]
    sorted_alayout = [x for _,x in sorted(zip(tests,alayout))]
    sorted_blayout = [x for _,x in sorted(zip(tests,blayout))]
    sorted_M = [x for _,x in sorted(zip(tests,M))]
    sorted_N = [x for _,x in sorted(zip(tests,N))]
    sorted_K = [x for _,x in sorted(zip(tests,K))]
    sorted_StrideA = [x for _,x in sorted(zip(tests,StrideA))]
    sorted_StrideB = [x for _,x in sorted(zip(tests,StrideB))]
    sorted_StrideC = [x for _,x in sorted(zip(tests,StrideC))]
    ck_gemm_params=[tlist,sorted_dtypes,sorted_alayout,sorted_blayout,
                sorted_M,sorted_N,sorted_K,sorted_StrideA,sorted_StrideB,
                sorted_StrideC]
    df=pd.DataFrame(np.transpose(ck_gemm_params),columns=['Test_number','Data_type',
        'Alayout','BLayout','M','N','K', 'StrideA','StrideB','StrideC'])
    print(df)

    dtypes = {
        'Test_number': Integer(),
        'Data_type': NVARCHAR(length=5),
        'Alayout': NVARCHAR(length=12),
        'Blayout': NVARCHAR(length=12),
        'M': Integer(),
        'N': Integer(),
        'K': Integer(),
        'StrideA': Integer(),
        'StrideB': Integer(),
        'StrideC': Integer()
        }
    df.to_sql("ck_gemm_test_params",connection,if_exists='replace',index=False, dtype=dtypes)
'''

def main():
    args = parse_args()
    results=[]
    tflops_base=[]
    testlist=[]
    #parse the test parameters from the logfile
    for filename in args.files:
        branch_name, node_id, gpu_arch, compute_units, rocm_vers, hip_vers, environment = get_log_params(filename)

    print("Branch name:",branch_name)
    print("Node name:",node_id)
    print("GPU_arch:",gpu_arch)
    print("Compute units:",compute_units)
    print("ROCM_version:",rocm_vers)
    print("HIP_version:",hip_vers)
    print("Environment:",environment)
    #parse results, get the Tflops value for "Best Perf" kernels
    results=parse_logfile(filename)

    print("Number of tests:",len(results))
    sql_hostname = '127.0.0.1'
    sql_username = os.environ["dbuser"]
    sql_password = os.environ["dbpassword"]
    sql_main_database = 'miopen_perf'
    sql_port = 3306
    ssh_host = os.environ["dbsship"]
    ssh_user = os.environ["dbsshuser"]
    ssh_port = int(os.environ["dbsshport"])
    ssh_pass = os.environ["dbsshpassword"]

    with SSHTunnelForwarder(
            (ssh_host, ssh_port),
            ssh_username=ssh_user,
            ssh_password=ssh_pass,
            remote_bind_address=(sql_hostname, sql_port)) as tunnel:

        sqlEngine = sqlalchemy.create_engine('mysql+pymysql://{0}:{1}@{2}:{3}/{4}'.
            format(sql_username, sql_password, sql_hostname, tunnel.local_bind_port, sql_main_database))
        conn = sqlEngine.connect()

        #save gemm performance tests:
        if 'perf_gemm.log' in filename:
            #write the ck_gemm_test_params table only needed once the test set changes
            #post_test_params(test_list,conn)
            for i in range(1,len(results)+1):
                testlist.append("Test%i"%i)
            table_name="ck_gemm_tflops"
        if 'batched_gemm' in filename:
            for i in range(1,len(results)+1):
                testlist.append("Test%i"%i)
            table_name="ck_batched_gemm_tflops"
        if 'grouped_gemm' in filename:
            for i in range(1,len(results)+1):
                testlist.append("Test%i"%i)
            table_name="ck_grouped_gemm_tflops"
        if 'conv_fwd' in filename:
            for i in range(1,len(results)+1):
                testlist.append("Test%i"%i)
            table_name="ck_conv_fwd_tflops"
        if 'conv_bwd_data' in filename:
            for i in range(1,len(results)+1):
                testlist.append("Test%i"%i)
            table_name="ck_conv_bwd_data_tflops"
        if 'gemm_bilinear' in filename:
            for i in range(1,len(results)+1):
                testlist.append("Test%i"%i)
            table_name="ck_gemm_bilinear_tflops"
        if 'reduction' in filename:
            for i in range(1,len(results)+1):
                testlist.append("Test%i"%i)
            table_name="ck_reduction_GBps"
        if 'resnet50_N4' in filename:
            for i in range(1,50):
                testlist.append("Layer%i"%i)
            table_name="ck_resnet50_N4_tflops"
        if 'resnet50_N256' in filename:
            for i in range(1,50):
                testlist.append("Layer%i"%i)
            table_name="ck_resnet50_N256_tflops"
        if 'onnx_gemm' in filename:
            for i in range(1,len(results)+1):
                testlist.append("Test%i"%i)
            table_name="ck_onnx_gemm_tflops"
        if 'splitK_gemm' in filename:
            for i in range(1,len(results)+1):
                testlist.append("Test%i"%i)
            table_name="ck_splitK_gemm_tflops"

        tflops_base = get_baseline(table_name,conn)
        store_new_test_result(table_name, results, testlist, branch_name, node_id, gpu_arch, compute_units, rocm_vers, hip_vers, environment, conn)
        conn.close()

    #compare the results to the baseline if baseline exists
    regression=0
    regression=compare_test_to_baseline(tflops_base,results,testlist)
    return regression

if __name__ == '__main__':
    main()
