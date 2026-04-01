//TEST:SIMPLE:

func foo(f : functype (float) -> int) -> int{
    return f(0);
}

int bit<T>(T) {
    return 1;
}

int bit<T, let N : int>(vector<T, N>) {
    return 1;
}

int zit() {
    // In an ideal world in this case we could infer that we want bit<T>
    return foo(bit<float>);
}
