//TEST:SIMPLE(filecheck=CHK):-target hlsl -entry main -stage vertex

void main()
{
  int2x2 a;
  int4 b;

  //CHK-NOT:error 39999: ambiguous call
  b.zw = mul(a, b.yx);
}
