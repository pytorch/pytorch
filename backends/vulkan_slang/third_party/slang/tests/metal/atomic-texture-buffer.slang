//TEST:SIMPLE(filecheck=METAL): -target metal -stage compute -entry computeMain
//TEST:SIMPLE(filecheck=METAL_FLOAT): -target metal -stage compute -entry computeMain -DFLOAT
//TEST:SIMPLE(filecheck=METALLIB): -target metallib -stage compute -entry computeMain

// METAL_FLOAT: floating point atomic operation

//METALLIB: @computeMain

RWBuffer<uint> uintBuffer;
RWBuffer<int> intBuffer;
RWBuffer<float> floatBuffer;

void test()
{
    int valInt = 1;
    int originalValueInt;
    int compareValueInt = 1;

    uint valUInt = 1;
    uint originalValueUInt;
    int compareValueUInt = 1;

    float valFloat = 1.0f;
    float originalValueFloat;

#ifdef FLOAT
    //float
	InterlockedExchange(floatBuffer[0], valFloat);
	InterlockedExchange(floatBuffer[0], valFloat, originalValueFloat);
#endif //FLOAT
 
    // buffer
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
	InterlockedAdd(intBuffer[0], valInt);
	InterlockedAnd(intBuffer[0], valInt);
	InterlockedMax(intBuffer[0], valInt);
	InterlockedMin(intBuffer[0], valInt);
	InterlockedOr(intBuffer[0], valInt);
	InterlockedXor(intBuffer[0], valInt);
	InterlockedAdd(intBuffer[0], valInt, originalValueInt);
	InterlockedAnd(intBuffer[0], valInt, originalValueInt);
	InterlockedMax(intBuffer[0], valInt, originalValueInt);
	InterlockedMin(intBuffer[0], valInt, originalValueInt);
	InterlockedOr(intBuffer[0], valInt, originalValueInt);
	InterlockedXor(intBuffer[0], valInt, originalValueInt);
	InterlockedExchange(intBuffer[0], valInt, originalValueInt);
	InterlockedCompareExchange(intBuffer[0], valInt, compareValueInt, originalValueInt);
	InterlockedCompareStore(intBuffer[0], valUInt, compareValueUInt);

	InterlockedAdd(uintBuffer[0], valUInt);
	InterlockedAnd(uintBuffer[0], valUInt);
	InterlockedMax(uintBuffer[0], valUInt);
	InterlockedMin(uintBuffer[0], valUInt);
	InterlockedOr(uintBuffer[0], valUInt);
	InterlockedXor(uintBuffer[0], valUInt);
	InterlockedAdd(uintBuffer[0], valUInt, originalValueUInt);
	InterlockedAnd(uintBuffer[0], valUInt, originalValueUInt);
	InterlockedMax(uintBuffer[0], valUInt, originalValueUInt);
	InterlockedMin(uintBuffer[0], valUInt, originalValueUInt);
	InterlockedOr(uintBuffer[0], valUInt, originalValueUInt);
	InterlockedXor(uintBuffer[0], valUInt, originalValueUInt);
	InterlockedExchange(uintBuffer[0], valUInt, originalValueUInt);
	InterlockedCompareExchange(uintBuffer[0], valUInt, compareValueUInt, originalValueUInt);
	InterlockedCompareStore(uintBuffer[0], valUInt, compareValueUInt);
}

[numthreads(1, 1, 1)]
void computeMain()
{
    test();
}
