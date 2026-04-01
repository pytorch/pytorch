// rewriter-parameter-block.slang
//TEST_IGNORE_FILE:

// A type that mixes uniform and resource fields
struct Data
{
	int val;
	RWStructuredBuffer<int> buf;
};

// A function that uses that type
int test(Data data, int val)
{
	return data.val + data.buf[val];
}

// A global-scope parameter block of the mixed type
ParameterBlock<Data> gA;

// A constant buffer declaration containing the mixed type
cbuffer C
{
	int extra;
	ParameterBlock<Data> gB;
};
