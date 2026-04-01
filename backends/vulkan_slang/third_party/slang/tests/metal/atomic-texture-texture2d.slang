//TEST:SIMPLE(filecheck=METAL): -target metal -stage compute -entry computeMain
//TEST:SIMPLE(filecheck=METALLIB): -target metallib -stage compute -entry computeMain

//METALLIB: @computeMain

RWTexture2D<uint> uintTexture2D;
RWTexture2D<int> intTexture2D;

RWTexture2DArray<uint> uintTexture2DArray;
RWTexture2DArray<int> intTexture2DArray;

void test()
{
    int valInt = 1;
    int originalValueInt;
    int compareValueInt = 1;

    uint valUInt = 1;
    uint originalValueUInt;
    int compareValueUInt = 1;

    // Texture2D
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
	InterlockedAdd(intTexture2D[0], valInt);
	InterlockedAnd(intTexture2D[0], valInt);
	InterlockedMax(intTexture2D[0], valInt);
	InterlockedMin(intTexture2D[0], valInt);
	InterlockedOr(intTexture2D[0], valInt);
	InterlockedXor(intTexture2D[0], valInt);
	InterlockedAdd(intTexture2D[0], valInt, originalValueInt);
	InterlockedAnd(intTexture2D[0], valInt, originalValueInt);
	InterlockedMax(intTexture2D[0], valInt, originalValueInt);
	InterlockedMin(intTexture2D[0], valInt, originalValueInt);
	InterlockedOr(intTexture2D[0], valInt, originalValueInt);
	InterlockedXor(intTexture2D[0], valInt, originalValueInt);
	InterlockedExchange(intTexture2D[0], valInt, originalValueInt);
	InterlockedCompareExchange(intTexture2D[0], valInt, compareValueInt, originalValueInt);
	InterlockedCompareStore(intTexture2D[0], valUInt, compareValueUInt);

	InterlockedAdd(uintTexture2D[0], valUInt);
	InterlockedAnd(uintTexture2D[0], valUInt);
	InterlockedMax(uintTexture2D[0], valUInt);
	InterlockedMin(uintTexture2D[0], valUInt);
	InterlockedOr(uintTexture2D[0], valUInt);
	InterlockedXor(uintTexture2D[0], valUInt);
	InterlockedAdd(uintTexture2D[0], valUInt, originalValueUInt);
	InterlockedAnd(uintTexture2D[0], valUInt, originalValueUInt);
	InterlockedMax(uintTexture2D[0], valUInt, originalValueUInt);
	InterlockedMin(uintTexture2D[0], valUInt, originalValueUInt);
	InterlockedOr(uintTexture2D[0], valUInt, originalValueUInt);
	InterlockedXor(uintTexture2D[0], valUInt, originalValueUInt);
	InterlockedExchange(uintTexture2D[0], valUInt, originalValueUInt);
	InterlockedCompareExchange(uintTexture2D[0], valUInt, compareValueUInt, originalValueUInt);
	InterlockedCompareStore(uintTexture2D[0], valUInt, compareValueUInt);

    // Texture2DArray
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
	InterlockedAdd(intTexture2DArray[0], valInt);
	InterlockedAnd(intTexture2DArray[0], valInt);
	InterlockedMax(intTexture2DArray[0], valInt);
	InterlockedMin(intTexture2DArray[0], valInt);
	InterlockedOr(intTexture2DArray[0], valInt);
	InterlockedXor(intTexture2DArray[0], valInt);
	InterlockedAdd(intTexture2DArray[0], valInt, originalValueInt);
	InterlockedAnd(intTexture2DArray[0], valInt, originalValueInt);
	InterlockedMax(intTexture2DArray[0], valInt, originalValueInt);
	InterlockedMin(intTexture2DArray[0], valInt, originalValueInt);
	InterlockedOr(intTexture2DArray[0], valInt, originalValueInt);
	InterlockedXor(intTexture2DArray[0], valInt, originalValueInt);
	InterlockedExchange(intTexture2DArray[0], valInt, originalValueInt);
	InterlockedCompareExchange(intTexture2DArray[0], valInt, compareValueInt, originalValueInt);
	InterlockedCompareStore(intTexture2DArray[0], valUInt, compareValueUInt);

	InterlockedAdd(uintTexture2DArray[0], valUInt);
	InterlockedAnd(uintTexture2DArray[0], valUInt);
	InterlockedMax(uintTexture2DArray[0], valUInt);
	InterlockedMin(uintTexture2DArray[0], valUInt);
	InterlockedOr(uintTexture2DArray[0], valUInt);
	InterlockedXor(uintTexture2DArray[0], valUInt);
	InterlockedAdd(uintTexture2DArray[0], valUInt, originalValueUInt);
	InterlockedAnd(uintTexture2DArray[0], valUInt, originalValueUInt);
	InterlockedMax(uintTexture2DArray[0], valUInt, originalValueUInt);
	InterlockedMin(uintTexture2DArray[0], valUInt, originalValueUInt);
	InterlockedOr(uintTexture2DArray[0], valUInt, originalValueUInt);
	InterlockedXor(uintTexture2DArray[0], valUInt, originalValueUInt);
	InterlockedExchange(uintTexture2DArray[0], valUInt, originalValueUInt);
	InterlockedCompareExchange(uintTexture2DArray[0], valUInt, compareValueUInt, originalValueUInt);
	InterlockedCompareStore(uintTexture2DArray[0], valUInt, compareValueUInt);
}

[numthreads(1, 1, 1)]
void computeMain()
{
    test();
}
