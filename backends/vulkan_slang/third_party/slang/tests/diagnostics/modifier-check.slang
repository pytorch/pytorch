//DIAGNOSTIC_TEST:SIMPLE(filecheck=CHECK):

// CHECK: ([[# @LINE+1]]): error 31201
layout(rgba8);
Texture2D<float4> tex;

// CHECK: ([[# @LINE+1]]): error 31202
public internal int v()
{}
