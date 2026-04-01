//TEST:INTERPRET(filecheck=BC): -disasm
//TEST:INTERPRET(filecheck=CHECK):

int main(int argc, NativeString* argv)
{
    printf("hello world\n");
    for (int i = 0; i < argc; ++i)
    {
        printf("%s\n", argv[i]);
    }
    return 100;
}

// CHECK: hello world
// CHECK: {{.*}}hello.slang
// BC: func main
// BC: ret