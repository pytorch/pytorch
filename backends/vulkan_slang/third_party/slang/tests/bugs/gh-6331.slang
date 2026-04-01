//TEST:SIMPLE(filecheck=CHECK): -target spirv
// CHECK: OpEntryPoint

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
[ForceInline]
__generic<T : __BuiltinFloatingPointType>
public T accumulateSample(uint64_t pSampleIdx, T pOldValue, T pNewValue) {
    T lTmp = isfinite(pOldValue) ? pOldValue : T(0);
    return lTmp + (pNewValue - lTmp) / T(pSampleIdx + 1);
}
[ForceInline]
__generic<T : __BuiltinFloatingPointType, let N : int>
public vector<T, N> accumulateSample(uint64_t pSampleIdx, vector<T, N> pOldValue, vector<T, N> pNewValue) {
    vector<T, N> lTmp = select(isfinite(pOldValue), pOldValue, T(0));
    return lTmp + (pNewValue - lTmp) / vector<T, N>(pSampleIdx + 1);
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
public struct RenderBuffer2D<T : ITexelElement> {
    public Texture2D<T> handle;
    public uint16_t2 offset;
    public uint16_t2 size;

    public __init() {
        handle = {};
        offset = {};
        size = {};
    }

    [ForceInline] public bool isValid() { return size[0] > 0u && size[1] > 0u; }
    [ForceInline] public uint64_t getPixelCount() { return (uint64_t)(size[0]) * (uint64_t)(size[1]); }
}

public struct RWRenderBuffer2D<T : ITexelElement> {
    public RWTexture2D<T> handle;
    public uint16_t2 offset;
    public uint16_t2 size;

    public __init() {
        handle = {};
        offset = {};
        size = {};
    }

    [ForceInline] public bool isValid() { return size[0] > 0u && size[1] > 0u; }
    [ForceInline] public uint64_t getPixelCount() { return (uint64_t)(size[0]) * (uint64_t)(size[1]); }
}

__generic<T : __BuiltinFloatingPointType>
public extension RWRenderBuffer2D<T> where T : ITexelElement {
    [ForceInline]
    public void accumulate(uint32_t2 pDestIdx, uint64_t pSampleIdx, T pSampleValue) {
        uint32_t2 lIndex = offset + pDestIdx;
        handle[lIndex] = accumulateSample(pSampleIdx, handle[lIndex], pSampleValue);
    }
}
__generic<T : __BuiltinFloatingPointType, let N : int>
public extension RWRenderBuffer2D<vector<T, N>> where vector<T,N>:ITexelElement {
    [ForceInline]
    public void accumulate(uint32_t2 pDestIdx, uint64_t pSampleIdx, vector<T, N> pSampleValue) {
        uint32_t2 lIndex = offset + pDestIdx;
        handle[lIndex] = accumulateSample(pSampleIdx, handle[lIndex], pSampleValue);
    }
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Global Parameters
public uniform uint32_t gSampleIdx;
public uniform RenderBuffer2D<float4> gInput;
public uniform RWRenderBuffer2D<float4> gOutput;
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
[shader("compute")]
[numthreads(16, 16, 1)]
void refineImage(uint3 pDispatchThreadIdx: SV_DispatchThreadID) {
    const uint2 lPixelCoordinates = { pDispatchThreadIdx.x, pDispatchThreadIdx.y };

    // Some sanity checks.
    if (!gInput.isValid() || any(lPixelCoordinates >= gInput.size)) {
        return;
    }
    if (!gOutput.isValid() || any(lPixelCoordinates >= gOutput.size)) {
        return;
    }

    gOutput.accumulate(lPixelCoordinates, gSampleIdx, gInput.handle[lPixelCoordinates]);
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
