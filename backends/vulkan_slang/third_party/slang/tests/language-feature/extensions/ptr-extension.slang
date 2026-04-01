//TEST:EXECUTABLE:
__generic<T> extension Ptr<T> {
  static func FromHandle(uint64_t handle) -> Ptr<T> {
    return (Ptr<T>)handle;
  }
}

__extern_cpp export
func main() -> int
{
    let x = Ptr<int>::FromHandle(0ull);
    printf("%d\n", (int)x);
    return 0;
}