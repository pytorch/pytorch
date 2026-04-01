//TEST:LANG_SERVER:
//HOVER:4,8

float dsqr<T:II

[ForwardDerivative(dsqr)]
float sqr<T:IInterface>(T obj, float x)
{
    return no_diff(obj.calc(x)) + x * x;
}
