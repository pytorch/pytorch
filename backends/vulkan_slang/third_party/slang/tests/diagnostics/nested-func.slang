//DIAGNOSTIC_TEST:SIMPLE(filecheck=CHECK):


struct VertexOutput{float Input;}
float4 fragmentMain(VertexOutput vertex) : SV_Target
{
   // CHECK: ([[# @LINE+1]]): error 30102
  static float GetValue(int val) { return vertex.Input; } // Simplified example
  float a = GetValue(vertex.Input);

  // etc
}
