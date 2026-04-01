module "that-module";

__include "that-module-impl";

void internalMethod();

public void publicMethod();

public namespace Namespace
{
    static int internalVar;

    public static int publicVar;
}

enum InternalEnum
{
    A,B,C
}
public enum PublicEnum
{
    D,E,F
}

struct InternalStruct { int x; }
