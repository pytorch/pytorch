//TEST:SIMPLE(filecheck=METAL): -target metal -stage compute -entry computeMain
//TEST:SIMPLE(filecheck=METALLIB): -target metallib -stage compute -entry computeMain

//METALLIB: @computeMain

RWTexture3D<uint> uintTexture3D;
RWTexture3D<int> intTexture3D;
void test()
{
    int valInt = 1;
    int originalValueInt;
    int compareValueInt = 1;

    uint valUInt = 1;
    uint originalValueUInt;
    int compareValueUInt = 1;

    // Texture3D
// METAL: .atomic_fetch_add
// METAL: .atomic_fetch_and
// METAL: .atomic_fetch_max
// METAL: .atomic_fetch_min
// METAL: .atomic_fetch_or
// METAL: .atomic_fetch_xor
// METAL: .atomic_fetch_add
// METAL: .atomic_fetch_and
// METAL: .atomic_fetch_max
// METAL: .atomic_fetch_min
// METAL: .atomic_fetch_or
// METAL: .atomic_fetch_xor
// METAL: .atomic_exchange
// METAL: .atomic_compare_exchange_weak
// METAL: .atomic_compare_exchange_weak

// METAL: .atomic_fetch_add
// METAL: .atomic_fetch_and
// METAL: .atomic_fetch_max
// METAL: .atomic_fetch_min
// METAL: .atomic_fetch_or
// METAL: .atomic_fetch_xor
// METAL: .atomic_fetch_add
// METAL: .atomic_fetch_and
// METAL: .atomic_fetch_max
// METAL: .atomic_fetch_min
// METAL: .atomic_fetch_or
// METAL: .atomic_fetch_xor
// METAL: .atomic_exchange
// METAL: .atomic_compare_exchange_weak
// METAL: .atomic_compare_exchange_weak
    InterlockedAdd(intTexture3D[0], valInt);
	InterlockedAnd(intTexture3D[0], valInt);
	InterlockedMax(intTexture3D[0], valInt);
	InterlockedMin(intTexture3D[0], valInt);
	InterlockedOr(intTexture3D[0], valInt);
	InterlockedXor(intTexture3D[0], valInt);
	InterlockedAdd(intTexture3D[0], valInt, originalValueInt);
	InterlockedAnd(intTexture3D[0], valInt, originalValueInt);
	InterlockedMax(intTexture3D[0], valInt, originalValueInt);
	InterlockedMin(intTexture3D[0], valInt, originalValueInt);
	InterlockedOr(intTexture3D[0], valInt, originalValueInt);
	InterlockedXor(intTexture3D[0], valInt, originalValueInt);
	InterlockedExchange(intTexture3D[0], valInt, originalValueInt);
	InterlockedCompareExchange(intTexture3D[0], valInt, compareValueInt, originalValueInt);
    InterlockedCompareStore(intTexture3D[0], valUInt, compareValueUInt);

	InterlockedAdd(uintTexture3D[0], valUInt);
	InterlockedAnd(uintTexture3D[0], valUInt);
	InterlockedMax(uintTexture3D[0], valUInt);
	InterlockedMin(uintTexture3D[0], valUInt);
	InterlockedOr(uintTexture3D[0], valUInt);
	InterlockedXor(uintTexture3D[0], valUInt);
	InterlockedAdd(uintTexture3D[0], valUInt, originalValueUInt);
	InterlockedAnd(uintTexture3D[0], valUInt, originalValueUInt);
	InterlockedMax(uintTexture3D[0], valUInt, originalValueUInt);
	InterlockedMin(uintTexture3D[0], valUInt, originalValueUInt);
	InterlockedOr(uintTexture3D[0], valUInt, originalValueUInt);
	InterlockedXor(uintTexture3D[0], valUInt, originalValueUInt);
	InterlockedExchange(uintTexture3D[0], valUInt, originalValueUInt);
	InterlockedCompareExchange(uintTexture3D[0], valUInt, compareValueUInt, originalValueUInt);
    InterlockedCompareStore(uintTexture3D[0], valUInt, compareValueUInt);

}

[numthreads(1, 1, 1)]
void computeMain()
{
    test();
}
