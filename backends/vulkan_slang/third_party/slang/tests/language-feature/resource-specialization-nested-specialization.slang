//TEST:SIMPLE(filecheck=CHECK_DXIL): -target dxil -profile sm_6_0 -entry computeMain -stage compute -DMEMBER_FUNCTION_CALL
//TEST:SIMPLE(filecheck=CHECK_DXIL): -target dxil -profile sm_6_0 -entry computeMain -stage compute

//CHECK_DXIL: computeMain

struct Grid
{
    uint bufSize;
    StructuredBuffer<uint> buf;
};

struct GridGeo
{
    Grid grids[2];

    void getGrid(uint index, out Grid grid)
    {
        grid = grids[index];
    }
};

struct Scene
{
    GridGeo gridGeo;

    void getGrid_BAD(uint index, out Grid grid)
    {
        gridGeo.getGrid(index, grid);
    }

    void getGrid_GOOD(uint index, out Grid grid)
    {
        grid = gridGeo.grids[index];
    }
};

ParameterBlock<Scene> gScene;
RWStructuredBuffer<uint> gridBuffers[2];
RWStructuredBuffer<uint> outputBuffer;

void direct_getGrid_BAD(uint index, out Grid grid)
{
    gScene.gridGeo.getGrid(index, grid);
}

[numthreads(1, 1, 1)]
void computeMain(uint3 threadId: SV_DispatchThreadID)
{
   
    Grid grid;

#ifdef MEMBER_FUNCTION_CALL
    direct_getGrid_BAD(1, grid);
#else 
    gScene.getGrid_BAD(1, grid);
#endif
    gridBuffers[0][1] = grid.buf[1];
}
