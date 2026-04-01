// pipeline-simple.slang.h


// TODO(tfoley): strip this down to a minimal pipeline

pipeline StandardPipeline
{
    [Pinned] input world MeshVertex;

    world CoarseVertex; // : "glsl(vertex:projCoord)" using projCoord export standardExport;
    world Fragment;     // : "glsl" export fragmentExport;

    require @CoarseVertex vec4 projCoord;

    [VertexInput] extern @CoarseVertex MeshVertex vertAttribIn;
    import(MeshVertex->CoarseVertex) vertexImport()
    {
        return project(vertAttribIn);
    }

    extern @Fragment CoarseVertex CoarseVertexIn;
    import(CoarseVertex->Fragment) standardImport()
    // TODO(tfoley): this trait doesn't seem to be implemented on `vec3`
    //        require trait IsTriviallyPassable(CoarseVertex)
    {
        return project(CoarseVertexIn);
    }

    stage vs : VertexShader
    {
    World:
        CoarseVertex;
    Position:
        projCoord;
    }

    stage fs : FragmentShader
    {
    World:
        Fragment;
    }
}