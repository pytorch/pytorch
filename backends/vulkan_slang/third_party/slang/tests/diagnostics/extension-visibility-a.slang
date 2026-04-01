// extension-visibility-a.slang

public interface IThing
{
	public int getValue();
}

// Note: not implementing the interface here!
public struct MyThing
{
	public int value;
}

public int helper<T : IThing>(T thing)
{
	return thing.getValue();
}
