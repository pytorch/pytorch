#!/usr/bin/env python3
import os, io, argparse, datetime, re
import numpy as np
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

def main():
    args = parse_args()
    tests = []
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
    #parse results, get the Tflops value for "Best Perf" kernels

    glue=""
    for filename in args.files:
        for line in open(filename):
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
            if 'InstalledDir' in line:
                lst=line.split()
                rocm_vers=lst[1][lst[1].find('/opt/rocm-')+len('/opt/rocm-'):lst[1].rfind('/llvm/bin')]
    print("Branch name:",branch_name)
    print("Node name:",node_id)
    print("GPU_arch:",gpu_arch)
    print("Compute units:",compute_units)
    print("ROCM_version:",rocm_vers)
    print("HIP_version:",hip_vers)


    #parse gemm performance tests:
    if 'gemm' in filename:
        for filename in args.files:
            for line in open(filename):
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
        #print("sorted tests:",sorted_tests)
        sorted_tflops = [x for _,x in sorted(zip(tests,tflops))]
        #sorted_kernels = [x for _,x in sorted(zip(tests,kernels))]
        test_list=list(range(1,len(tests)+1))

    #parse resnet50 performance tests:
    if 'resnet50' in filename:
        for filename in args.files:
            for line in open(filename):
                if 'Best Perf' in line:
                    lst=line.split()
                    tflops.append(lst[4])

    print("Number of tests:",len(tflops))
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
        if 'gemm' in filename:

            #write the ck_gemm_test_params table
            #only needed once the test set changes
            '''
            sorted_dtypes = [x for _,x in sorted(zip(tests,dtype))]
            sorted_alayout = [x for _,x in sorted(zip(tests,alayout))]
            sorted_blayout = [x for _,x in sorted(zip(tests,blayout))]
            sorted_M = [x for _,x in sorted(zip(tests,M))]
            sorted_N = [x for _,x in sorted(zip(tests,N))]
            sorted_K = [x for _,x in sorted(zip(tests,K))]
            sorted_StrideA = [x for _,x in sorted(zip(tests,StrideA))]
            sorted_StrideB = [x for _,x in sorted(zip(tests,StrideB))]
            sorted_StrideC = [x for _,x in sorted(zip(tests,StrideC))]
            ck_gemm_params=[test_list,sorted_dtypes,sorted_alayout,sorted_blayout,
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
            df.to_sql("ck_gemm_test_params",conn,if_exists='replace',index=False, dtype=dtypes)
            '''

            #read baseline results for the latest develop branch
            query = '''SELECT * from ck_gemm_tflops WHERE Datetime = (SELECT MAX(Datetime) FROM ck_gemm_tflops where Branch_ID='develop' );'''
            tflops_base = pd.read_sql_query(query, conn)

            #write new results to the db
            testlist=[]
            for i in range(1,len(tests)+1):
                testlist.append("Test%i"%i)
            ck_gemm_tflops=[str(branch_name),str(node_id),str(gpu_arch),compute_units,str(rocm_vers),str(hip_vers),str(datetime.datetime.now())]
            flops=pd.DataFrame(data=[ck_gemm_tflops],columns=['Branch_ID','Node_ID','GPU_arch','Compute Units','ROCM_version','HIP_version','Datetime'])
            df_add=pd.DataFrame(data=[sorted_tflops],columns=testlist)
            flops=pd.concat([flops,df_add],axis=1)
            print("new tflops for gemm tests:",flops)
            flops.to_sql("ck_gemm_tflops",conn,if_exists='append',index=False)

        #save resnet50 performance tests:
        if 'resnet50' in filename:
            #read baseline results for the latest develop branch
            query = '''SELECT * from ck_resnet50_N256_tflops WHERE Datetime = (SELECT MAX(Datetime) FROM ck_resnet50_N256_tflops where Branch_ID='develop' );'''
            tflops_base_N256 = pd.read_sql_query(query, conn)
            query = '''SELECT * from ck_resnet50_N4_tflops WHERE Datetime = (SELECT MAX(Datetime) FROM ck_resnet50_N4_tflops where Branch_ID='develop' );'''
            tflops_base_N4 = pd.read_sql_query(query, conn)

            #write new results to the db
            testlist=[]
            for i in range(1,50):
                testlist.append("Layer%i"%i)
            ck_resnet_tflops=[str(branch_name),str(node_id),str(gpu_arch),compute_units,str(rocm_vers),str(hip_vers),str(datetime.datetime.now())]
            flops0=pd.DataFrame(data=[ck_resnet_tflops],columns=['Branch_ID','Node_ID','GPU_arch','Compute Units','ROCM_version','HIP_version','Datetime'])
            df_add=pd.DataFrame(data=[tflops[0:49]],columns=testlist)
            flops=pd.concat([flops0,df_add],axis=1)
            print("new tflops for N=256 resnet50 test:",flops)
            flops.to_sql("ck_resnet50_N256_tflops",conn,if_exists='append',index=False)
            df_add=pd.DataFrame(data=[tflops[49:98]],columns=testlist)
            flops=pd.concat([flops0,df_add],axis=1)
            print("new tflops for N=4 resnet50 test:",flops)
            flops.to_sql("ck_resnet50_N4_tflops",conn,if_exists='append',index=False)

        conn.close()

    #compare the results to the baseline if baseline exists
    regression=0
    if 'gemm' in filename:
        if not tflops_base.empty:
            base=tflops_base[testlist].to_numpy(dtype='float')
            base_list=base[0]
            ave_perf=0
            for i in range(len(base_list)):
                # success criterion:
                if base_list[i]>1.01*float(sorted_tflops[i]):
                    print("test # ",i,"shows regression by {:.3f}%".format(
                        (float(sorted_tflops[i])-base_list[i])/base_list[i]*100))
                    regression=1
                ave_perf=ave_perf+float(sorted_tflops[i])/base_list[i]
            if regression==0:
                print("no regressions found")
            ave_perf=ave_perf/len(base_list)
            print("average performance relative to baseline:",ave_perf)
        else:
            print("could not find a baseline")
    if 'resnet50' in filename:
        if not tflops_base_N256.empty:
            base=tflops_base_N256[testlist].to_numpy(dtype='float')
            base_list=base[0]
            ave_perf=0
            for i in range(len(base_list)):
                # success criterion:
                if base_list[i]>1.01*float(tflops[i]):
                    print("layer # ",i,"shows regression by {:.3f}%".format(
                        (float(tflops[i])-base_list[i])/base_list[i]*100))
                    regression=1
                ave_perf=ave_perf+float(tflops[i])/base_list[i]
            if regression==0:
                print("no regressions found")
            ave_perf=ave_perf/len(base_list)
            print("average performance relative to baseline:",ave_perf)
        else:
            print("could not find a baseline for N=256")
        if not tflops_base_N4.empty:
            base=tflops_base_N4[testlist].to_numpy(dtype='float')
            base_list=base[0]
            ave_perf=0
            for i in range(len(base_list)):
                # success criterion:
                if base_list[i]>1.01*float(tflops[i+49]):
                    print("layer # ",i,"shows regression by {:.3f}%".format(
                        (float(tflops[i+49])-base_list[i])/base_list[i]*100))
                    regression=1
                ave_perf=ave_perf+float(tflops[i+49])/base_list[i]
            if regression==0:
                print("no regressions found")
            ave_perf=ave_perf/len(base_list)
            print("average performance relative to baseline:",ave_perf)
        else:
            print("could not find a baseline for N=4")

    #return 0 if performance criteria met, otherwise return 1
    return regression

if __name__ == '__main__':
    main()