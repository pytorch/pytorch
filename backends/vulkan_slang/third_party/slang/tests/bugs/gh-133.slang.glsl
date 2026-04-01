#version 420
//TEST_IGNORE_FILE:

struct Fragment
{
	uint foo;
};

layout(binding = 0)
uniform U
{
	uint bar;
};

Fragment main_()
{
	Fragment result;
	result.foo = bar;
	return result;
}

layout(location = 0)
out uint SLANG_out_main_result_foo;

void main()
{
	Fragment main_result = main_();
	SLANG_out_main_result_foo = main_result.foo;
}
