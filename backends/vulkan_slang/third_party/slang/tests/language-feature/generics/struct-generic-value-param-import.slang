// struct-generic-value-param-import.slang
//TEST_IGNORE_FILE:

// This file is used by `struct-generic-value-param.slang`,
// and also incidentally tests that a trailing `;` is optional
// for `struct` decalrations in Slang files, including
// any `import`ed code.

public interface IData {}

public struct Data<let kCount : int> : IData
{
    public int state;

    [mutating] public void doStuff()
    {
        state++;
    }
}
