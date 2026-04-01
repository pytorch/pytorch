//TEST(compute, vulkan):SIMPLE(filecheck=CHECK_SPV): -stage compute -entry computeMain -target spirv -emit-spirv-directly -g2


// CHECK_SPV: [[STRING_uint:%[1-9][0-9]*]] = OpString "uint"
// CHECK_SPV: [[TYPE_uint:%[1-9][0-9]*]] = OpExtInst %void %{{[0-9]*}} DebugTypeBasic [[STRING_uint]]

// Debug type qualifier for atomic is 3
// CHECK_SPV: [[TYPE_atomic_uint:%[1-9][0-9]*]] = OpExtInst %void %{{[0-9]*}} DebugTypeQualifier [[TYPE_uint]] %uint_3
// CHECK_SPV: OpExtInst %void %{{[0-9]*}} DebugTypePointer [[TYPE_atomic_uint]]

struct Test {
    Atomic<uint32_t> atomicMember;
};

struct PC {
    Atomic<uint>* atomicMember;
    Test* test;
}

[[vk::push_constant]]
PC pc;

[shader("compute")]
[numthreads(1,1,1)]
public void computeMain() {
    Atomic<uint32_t>* atomicMember = pc.atomicMember;
    atomicMember.store(10);

    Test* test = pc.test;
    test.atomicMember.store(10);
}
