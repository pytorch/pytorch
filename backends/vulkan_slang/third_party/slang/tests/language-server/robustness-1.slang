//TEST:LANG_SERVER:
//
// tests empty subscript argument.

void main()
{
//HOVER:8,13
    float3 arr[3];
    arr[].;

}
