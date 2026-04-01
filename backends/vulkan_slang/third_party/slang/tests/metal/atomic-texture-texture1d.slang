//TEST:SIMPLE(filecheck=METAL): -target metal -stage compute -entry computeMain
//TEST:SIMPLE(filecheck=METALLIB): -target metallib -stage compute -entry computeMain

//METALLIB: @computeMain

RWTexture1D<uint> uintTexture1D;
RWTexture1D<int> intTexture1D;

RWTexture1DArray<uint> uintTexture1DArray;
RWTexture1DArray<int> intTexture1DArray;

void test()
{
    int valInt = 1;
    int originalValueInt;
    int compareValueInt = 1;

    uint valUInt = 1;
    uint originalValueUInt;
    int compareValueUInt = 1;

    // Texture1D
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
	InterlockedAdd(intTexture1D[0], valInt);
	InterlockedAnd(intTexture1D[0], valInt);
	InterlockedMax(intTexture1D[0], valInt);
	InterlockedMin(intTexture1D[0], valInt);
	InterlockedOr(intTexture1D[0], valInt);
	InterlockedXor(intTexture1D[0], valInt);
	InterlockedAdd(intTexture1D[0], valInt, originalValueInt);
	InterlockedAnd(intTexture1D[0], valInt, originalValueInt);
	InterlockedMax(intTexture1D[0], valInt, originalValueInt);
	InterlockedMin(intTexture1D[0], valInt, originalValueInt);
	InterlockedOr(intTexture1D[0], valInt, originalValueInt);
	InterlockedXor(intTexture1D[0], valInt, originalValueInt);
	InterlockedExchange(intTexture1D[0], valInt, originalValueInt);
	InterlockedCompareExchange(intTexture1D[0], valInt, compareValueInt, originalValueInt);
	InterlockedCompareStore(intTexture1D[0], valUInt, compareValueUInt);

	InterlockedAdd(uintTexture1D[0], valUInt);
	InterlockedAnd(uintTexture1D[0], valUInt);
	InterlockedMax(uintTexture1D[0], valUInt);
	InterlockedMin(uintTexture1D[0], valUInt);
	InterlockedOr(uintTexture1D[0], valUInt);
	InterlockedXor(uintTexture1D[0], valUInt);
	InterlockedAdd(uintTexture1D[0], valUInt, originalValueUInt);
	InterlockedAnd(uintTexture1D[0], valUInt, originalValueUInt);
	InterlockedMax(uintTexture1D[0], valUInt, originalValueUInt);
	InterlockedMin(uintTexture1D[0], valUInt, originalValueUInt);
	InterlockedOr(uintTexture1D[0], valUInt, originalValueUInt);
	InterlockedXor(uintTexture1D[0], valUInt, originalValueUInt);
	InterlockedExchange(uintTexture1D[0], valUInt, originalValueUInt);
	InterlockedCompareExchange(uintTexture1D[0], valUInt, compareValueUInt, originalValueUInt);
	InterlockedCompareStore(uintTexture1D[0], valUInt, compareValueUInt);

    // Texture1DArray
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
	InterlockedAdd(intTexture1DArray[0], valInt);
	InterlockedAnd(intTexture1DArray[0], valInt);
	InterlockedMax(intTexture1DArray[0], valInt);
	InterlockedMin(intTexture1DArray[0], valInt);
	InterlockedOr(intTexture1DArray[0], valInt);
	InterlockedXor(intTexture1DArray[0], valInt);
	InterlockedAdd(intTexture1DArray[0], valInt, originalValueInt);
	InterlockedAnd(intTexture1DArray[0], valInt, originalValueInt);
	InterlockedMax(intTexture1DArray[0], valInt, originalValueInt);
	InterlockedMin(intTexture1DArray[0], valInt, originalValueInt);
	InterlockedOr(intTexture1DArray[0], valInt, originalValueInt);
	InterlockedXor(intTexture1DArray[0], valInt, originalValueInt);
	InterlockedExchange(intTexture1DArray[0], valInt, originalValueInt);
	InterlockedCompareExchange(intTexture1DArray[0], valInt, compareValueInt, originalValueInt);
	InterlockedCompareStore(intTexture1DArray[0], valUInt, compareValueUInt);

	InterlockedAdd(uintTexture1DArray[0], valUInt);
	InterlockedAnd(uintTexture1DArray[0], valUInt);
	InterlockedMax(uintTexture1DArray[0], valUInt);
	InterlockedMin(uintTexture1DArray[0], valUInt);
	InterlockedOr(uintTexture1DArray[0], valUInt);
	InterlockedXor(uintTexture1DArray[0], valUInt);
	InterlockedAdd(uintTexture1DArray[0], valUInt, originalValueUInt);
	InterlockedAnd(uintTexture1DArray[0], valUInt, originalValueUInt);
	InterlockedMax(uintTexture1DArray[0], valUInt, originalValueUInt);
	InterlockedMin(uintTexture1DArray[0], valUInt, originalValueUInt);
	InterlockedOr(uintTexture1DArray[0], valUInt, originalValueUInt);
	InterlockedXor(uintTexture1DArray[0], valUInt, originalValueUInt);
	InterlockedExchange(uintTexture1DArray[0], valUInt, originalValueUInt);
	InterlockedCompareExchange(uintTexture1DArray[0], valUInt, compareValueUInt, originalValueUInt);
	InterlockedCompareStore(uintTexture1DArray[0], valUInt, compareValueUInt);
}

[numthreads(1, 1, 1)]
void computeMain()
{
    test();
}
