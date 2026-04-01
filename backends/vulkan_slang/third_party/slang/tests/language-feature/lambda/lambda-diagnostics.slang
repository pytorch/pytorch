//TEST:SIMPLE(filecheck=CHECK): -target spirv

[numthreads(1,1,1)]
void test()
{
    int c = 0;
    let lam = (int a) => {
        //CHECK:([[# @LINE+1]]): error 30011:
        c = a + 1;
    };
    let lam1 = (int a) => {
        if (a > 0)
            return 1;
        else
            //CHECK:([[# @LINE+1]]): error 30111:
            return 3.0;
    };
    RayQuery q;
    let lam2 = (int a) => {
        //CHECK:([[# @LINE+1]]): error 30112:
        q.Abort();
    };
}