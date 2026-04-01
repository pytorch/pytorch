implementing fcpw;
__include bounding_volumes;

public interface IBvhNode
{
    // returns the bounding box of the node
    BoundingBox getBoundingBox();

    // sets the bounding box of the node
    [mutating]
    void setBoundingBox(BoundingBox box_);

    // checks if the node has a bounding cone
    bool hasBoundingCone();

    // returns the bounding cone of the node
    BoundingCone getBoundingCone();

    // sets the bounding cone of the node
    [mutating]
    void setBoundingCone(BoundingCone cone_);

    // checks if the node is a leaf node
    bool isLeaf();

    // returns the offset to the right child of the interior node
    uint getRightChildOffset();

    // returns the number of primitives in the node
    uint getNumPrimitives();

    // returns the offset to the first primitive of the leaf node
    uint getPrimitiveOffset();

    // returns the number of silhouettes in the node
    uint getNumSilhouettes();

    // returns the offset to the first silhouette of the leaf node
    uint getSilhouetteOffset();
};

public struct BvhNode : IBvhNode
{
    public BoundingBox box;
    public uint nPrimitives;
    public uint offset;

    // returns the bounding box of the node
    public BoundingBox getBoundingBox()
    {
        return box;
    }

    // sets the bounding box of the node
    [mutating]
    public void setBoundingBox(BoundingBox box_)
    {
        box = box_;
    }

    // checks if the node has a bounding cone
    public bool hasBoundingCone()
    {
        return false;
    }

    // returns the bounding cone of the node
    public BoundingCone getBoundingCone()
    {
        return BoundingCone();
    }

    // sets the bounding cone of the node
    [mutating]
    public void setBoundingCone(BoundingCone cone_)
    {
        // do nothing
    }

    // checks if the node is a leaf node
    public bool isLeaf()
    {
        return nPrimitives > 0;
    }

    // returns the offset to the right child of the interior node
    public uint getRightChildOffset()
    {
        return offset;
    }

    // returns the number of primitives in the node
    public uint getNumPrimitives()
    {
        return nPrimitives;
    }

    // returns the offset to the first primitive of the leaf node
    public uint getPrimitiveOffset()
    {
        return offset;
    }

    // returns the number of silhouettes in the node
    public uint getNumSilhouettes()
    {
        return 0;
    }

    // returns the offset to the first silhouette of the leaf node
    public uint getSilhouetteOffset()
    {
        return 0;
    }
};

public struct SnchNode : IBvhNode
{
    public BoundingBox box;
    public BoundingCone cone;
    public uint nPrimitives;
    public uint offset;
    public uint nSilhouettes;
    public uint silhouetteOffset;

    // returns the bounding box of the node
    public BoundingBox getBoundingBox()
    {
        return box;
    }

    // sets the bounding box of the node
    [mutating]
    public void setBoundingBox(BoundingBox box_)
    {
        box = box_;
    }

    // checks if the node has a bounding cone
    public bool hasBoundingCone()
    {
        return true;
    }

    // returns the bounding cone of the node
    public BoundingCone getBoundingCone()
    {
        return cone;
    }

    // sets the bounding cone of the node
    [mutating]
    public void setBoundingCone(BoundingCone cone_)
    {
        cone = cone_;
    }

    // checks if the node is a leaf node
    public bool isLeaf()
    {
        return nPrimitives > 0;
    }

    // returns the offset to the right child of the interior node
    public uint getRightChildOffset()
    {
        return offset;
    }

    // returns the number of primitives in the node
    public uint getNumPrimitives()
    {
        return nPrimitives;
    }

    // returns the offset to the first primitive of the leaf node
    public uint getPrimitiveOffset()
    {
        return offset;
    }

    // returns the number of silhouettes in the node
    public uint getNumSilhouettes()
    {
        return nSilhouettes;
    }

    // returns the offset to the first silhouette of the leaf node
    public uint getSilhouetteOffset()
    {
        return silhouetteOffset;
    }
};
