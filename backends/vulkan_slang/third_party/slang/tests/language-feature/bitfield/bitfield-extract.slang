//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=CHECK):-vk -compile-arg -skip-spirv-validation -emit-spirv-directly
//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=CHECK):-dx12 -use-dxil
//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=CHECK):-mtl
//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=CHECK):-cpu
//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=CHECK):-cuda
//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=CHECK):-wgpu

// CHECK: 1
// CHECK-NEXT: 2
// CHECK-NEXT: 3
// CHECK-NEXT: 4
// CHECK-NEXT: 5
// CHECK-NEXT: 6
// CHECK-NEXT: 7
// CHECK-NEXT: 8
// CHECK-NEXT: 21
// CHECK-NEXT: 7A
// CHECK-NEXT: FFFFFFFA
// CHECK-NEXT: A
// CHECK-NEXT: 67
// CHECK-NEXT: FFFFFFEF
// CHECK-NEXT: 32
// CHECK-NEXT: FFFFFFA9

//TEST_INPUT:ubuffer(data=[-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1], stride=4):out,name=i32Buffer
RWStructuredBuffer<int> i32Buffer;

[numthreads(1, 1, 1)]
void computeMain()
{
    // 32-bit tests
    {
        // Simple hex extraction to test, varying the offset.
        uint value = 0x87654321;
        i32Buffer[0] = bitfieldExtract(value, 4 * 0, 4);
        i32Buffer[1] = bitfieldExtract(value, 4 * 1, 4);
        i32Buffer[2] = bitfieldExtract(value, 4 * 2, 4);
        i32Buffer[3] = bitfieldExtract(value, 4 * 3, 4);
        i32Buffer[4] = bitfieldExtract(value, 4 * 4, 4);
        i32Buffer[5] = bitfieldExtract(value, 4 * 5, 4);
        i32Buffer[6] = bitfieldExtract(value, 4 * 6, 4);
        i32Buffer[7] = bitfieldExtract(value, 4 * 7, 4);

        // Now varying the bit length
        value = 0b00111011111011110001111010100001;
        i32Buffer[8] = bitfieldExtract(value, 0, 6);
        i32Buffer[9] = bitfieldExtract(value, 6, 8);

        // Sign extension case
        //  - For unsigned data types, the most significant bits of the result will be set to zero.
        //  - For signed data types, the most significant bits will be set to the value of bit offset + base - 1
        //    (i.e., it is sign extended to the width of the return type).
        i32Buffer[10] = bitfieldExtract(0b1010111, 3, 4);  // 0b1010 -> 0b11111111111111111111111111111010
        i32Buffer[11] = bitfieldExtract(0b1010111u, 3, 4); // 0b1111 -> 0b00000000000000000000000000001010

        // Component-wise extraction
        int4 val4 = int4(0x12345678, 0x9abcdef0, 0x87654321, 0xfedcba98);
        int4 ext4 = bitfieldExtract(val4, 4, 8);
        i32Buffer[12] = ext4.x;
        i32Buffer[13] = ext4.y;
        i32Buffer[14] = ext4.z;
        i32Buffer[15] = ext4.w;
    }
}
