//TEST:LANG_SERVER(filecheck=CHECK):
module module_A;
struct A {
    int x;
    int y;
};

[numthreads(1,1,1)]
void main()
{
//HOVER:12,5
    A a; 
    a.x = 2;
}

// CHECK: struct A