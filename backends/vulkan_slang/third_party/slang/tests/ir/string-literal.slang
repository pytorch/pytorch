//TEST(compute):SIMPLE(filecheck=CHECK):-dump-ir -stage compute -entry main

// CHECK: global_hashed_string_literals("Hello \t\n\0x083 World")
[numthreads(1, 1, 1)]
void main(
    uint tid   : SV_DispatchThreadID)
{
	"Hello \t\n\0x083 World";
}

