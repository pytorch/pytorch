// x-macro.slang
//TEST:SIMPLE:

// Test the case of an "X macro" that takes another macro as a parameter

#define X(M) M(0) M(1) M(2) M(3) M(4) M(5) M(6) M(7)

#define A(x) + x + x + x

int sum()
{
	return X(A);
}
