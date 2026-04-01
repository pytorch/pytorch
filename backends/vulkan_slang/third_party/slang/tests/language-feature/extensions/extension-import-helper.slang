// extension-import-helper.slang

//TEST_IGNORE_FILE:

public interface IThing
{
	public float getValue();
}

public extension float : IThing
{
	public float getValue() { return this; }
}
