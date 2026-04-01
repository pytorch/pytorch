// x-macro-line-continuation.slang
//TEST:SIMPLE:

// Tests line continuations on diagnostic output of macros

#define X(M) \
    M(0) \
    M(1) \
    M(2) \
    M(3) \
    M(4, 4) \
    M(5) \
    M(6) \
    M(7)

#define A(x) + x + x + x

int sum()
{
	return X(A);
}
