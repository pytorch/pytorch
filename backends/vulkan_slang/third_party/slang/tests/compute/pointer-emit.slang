//TEST(compute):COMPARE_COMPUTE:-cpu -shaderobj
// Test emitting logic of pointer types.


struct Something
{
	int4 field0;
	int field1;
};

struct ArrayType
{
	Something data[2];
};

ArrayType getArray()
{
	ArrayType rs;
	rs.data[0].field0 = int4(1,2,3,4);
	rs.data[0].field1 = 1;
	rs.data[1].field0 = int4(2,3,4,5);
	rs.data[1].field1 = 2;
	return rs;
}

int inoutFunc(in out int param, 
			  int[2] arrayParam,
			  out int[2] arrayOut,
			  Something s0,
			  out Something s1)
{
    int v = 1;
    param = v + 1;

	arrayOut[0] = arrayParam[0];
	arrayOut[1] = arrayParam[1];

	s1.field1 = s0.field1 + 1;

    return param;
}

static int globalVar = 1;

int test(int inVal)
{
    int p = inVal;
	int t1[2] = {1, 2};
	int t2[2] = {0, 0};
	Something s, s_out;
	s.field1 = -1;
	s.field1 = s.field1 - 1;
	s.field0[0] = 1;
	int4 localVector = int4(1,2,3,4);
	localVector.x = 2;
	var array = getArray();
	return inoutFunc(p, t1, t2, s, s_out) + t2[0] + s_out.field1 +
		s.field0[0] + localVector.x + array.data[1].field1 + globalVar;
}

//TEST_INPUT:ubuffer(data=[0 1 2 3], stride=4):out,name=outputBuffer
RWStructuredBuffer<int> outputBuffer : register(u0);

[numthreads(4, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
	uint tid = dispatchThreadID.x;
	int inVal = outputBuffer[tid];
	int outVal = test(inVal);
	outputBuffer[tid] = outVal;
}
