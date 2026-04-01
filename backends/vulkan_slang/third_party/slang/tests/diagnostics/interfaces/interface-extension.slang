//DIAGNOSTIC_TEST:SIMPLE(filecheck=CHECK):-target cpp -stage compute -entry main -disable-specialization

interface IFoo{}


// CHECK: ([[# @LINE+1]]): error 30852
extension IFoo
{
    int f() { return 0; }
}