#pragma once

TYPED_TEST(TestSoftmax, ReduceOutermostDim)
{
    std::vector<ck::index_t> reduce_dims{this->Rank - 1};
    this->Run(reduce_dims);
}

TYPED_TEST(TestSoftmax, ReduceMiddleDim)
{
    for(int dim = 0; dim < this->Rank - 1; ++dim)
    {
        std::vector<ck::index_t> reduce_dims{dim};
        this->Run(reduce_dims);
    }
}

TYPED_TEST(TestSoftmax, ReduceMultipleDimsWithOutermost)
{
    for(int dim = 0; dim < this->Rank - 1; ++dim)
    {
        std::vector<ck::index_t> reduce_dims{dim, this->Rank - 1};
        this->Run(reduce_dims);
    }
}

TYPED_TEST(TestSoftmax, ReduceMultipleMiddleDims)
{
    std::vector<ck::index_t> reduce_dims{0, 1};
    if(this->Rank >= 3)
    {
        this->Run(reduce_dims);
    }

    if(this->Rank >= 4)
    {
        reduce_dims = std::vector<ck::index_t>{0, 2};
        this->Run(reduce_dims);
        reduce_dims = std::vector<ck::index_t>{0, 1, 2};
        this->Run(reduce_dims);
    }
}

TYPED_TEST(TestSoftmax, ReduceAllDims)
{
    std::vector<ck::index_t> reduce_dims(this->Rank);
    std::iota(std::begin(reduce_dims), std::end(reduce_dims), 0);
    this->Run(reduce_dims);
}

TYPED_TEST(TestSoftmax, ReduceOddLengths)
{
    this->in_lengths_ = {{3, 63, 1032}};
    if(this->Rank >= 4)
    {
        this->in_lengths_ = {{1, 3, 63, 1032}};
    }
    this->Run({this->Rank - 1});
    this->Run({this->Rank - 2});
}
