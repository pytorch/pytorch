//TEST:COMPARE_COMPUTE(filecheck-buffer=CHECK): -output-using-type

interface IFoo<T> {
    associatedtype Output : IBar;
    func foo(other: T) -> Output;
}

interface IBar { int getId(); }

struct Ant:IBar { int getId() { return 0; } };
struct Bat:IBar { int getId() { return 1; } };
struct Cat:IBar { int getId() { return 2; } };
struct Dog:IBar { int getId() { return 3; } };
struct Ewe:IBar { int getId() { return 4; } };
struct Fox:IBar { int getId() { return 5; } };
struct Gnu:IBar { int getId() { return 6; } };

extension Ant: IFoo<Bat> {
    typedef Cat Output;
    func foo(other: Bat) -> Cat { return Cat(); }
}
extension Ant: IFoo<Dog> {
    typedef Ewe Output;
    func foo(other: Dog) -> Ewe { return Ewe(); }
}
extension Ant: IFoo<Fox> {
    typedef Gnu Output;
    func foo(other: Fox) -> Gnu { return Gnu(); }
}

int test<T:IFoo<Fox>>(T v) {
    return v.foo(Fox()).getId();
}

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=output
RWStructuredBuffer<int> output;

[numthreads(1,1,1)]
void computeMain() {
    Ant a;
    // CHECK: 6
    output[0] = test(a);
}