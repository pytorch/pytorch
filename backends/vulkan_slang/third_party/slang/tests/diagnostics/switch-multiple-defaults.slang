//TEST:SIMPLE(filecheck=CHECK):

// Test to evaluate the behavior of unreachable code blocks within a switch statement. A switch statement with multiple default cases is not allowed and should throw an error

void test()
{
    switch (0)
    {
        default: break;
        // CHECK: ([[# @LINE+1]]): error 30600: {{.*}}
        default: break;
    }
	return;
}
