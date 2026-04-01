//TEST:SIMPLE(filecheck=CHECK):

// Tests to evaluate the behavior of code blocks within a switch statement. A switch statement with duplicate cases with same values is not allowed and should throw an error

enum class Cases
{
    A,
    B
};

void test1(Cases c)
{
    switch (c)
    {
        case Cases::A: break;
        case Cases::B: break;
        // CHECK: ([[# @LINE+1]]): error 30601: {{.*}}
        case Cases::A: break;
    }
    return;
}

void test2()
{
    switch (0)
    {
        case 1: break;
        case 2: break;
        // CHECK: ([[# @LINE+1]]): error 30601: {{.*}}
        case 1: break;
    }
	return;
}
