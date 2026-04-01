// extension-full-name.slang
//DIAGNOSTIC_TEST:SIMPLE:-target hlsl

// Define a generic type with a nested struct
struct GenericType<T>
{
    T value;

    struct InnerType
    {
        T innerValue;
    }

    InnerType inner;
}

// Define a non-generic type
struct NonGenericType
{
    float value;

    struct InnerType
    {
        float innerValue;
    }

    InnerType inner;
}

// Add an extension to the generic type
extension GenericType<half>
{
    // Type in extension
    struct State
    {
        half value;
        half factor;
    }

    // Function in extension
    State
    createState()
    {
        State state;
        state.value = value;
        state.factor = 1.0h;
        return state;
    }

    // Value/field in extension
    static State defaultState;
}

// Add an extension to a nested type inside a generic type
extension GenericType<float>.InnerType
{
    struct Options
    {
        float min;
        float max;
    }

    Options
    getOptions()
    {
        Options opts;
        opts.min = 0;
        opts.max = innerValue;
        return opts;
    }

    static Options defaultOptions;
}

// Add an extension to the non-generic type
extension NonGenericType
{
    // Type in extension
    struct Config
    {
        float threshold;
        float scale;
    }

    // Function in extension
    Config
    createConfig()
    {
        Config config;
        config.threshold = 0.5;
        config.scale = 2.0;
        return config;
    }

    // Value/field in extension
    static Config defaultConfig;
}

// Add nested types and extensions
struct Container
{
    struct Nested
    {
        float value;

        struct DeepNested
        {
            int count;
        }

        DeepNested deep;
    }

    Nested nested;
}

// Extension for deeply nested type
extension Container.Nested.DeepNested
{
    struct Record
    {
        int id;
        float value;
    }

    Record
    createRecord(int newId)
    {
        Record r;
        r.id = newId;
        r.value = 0;
        return r;
    }

    static Record defaultRecord;
}

extension Container.Nested
{
    // Type in nested extension
    struct Settings
    {
        float value;
        float threshold;
    }

    // Function in nested extension
    Settings
    createSettings()
    {
        Settings settings;
        settings.value = value;
        settings.threshold = 0.1;
        return settings;
    }

    // Value/field in nested extension
    static Settings defaultSettings;
}

[shader("compute")][numthreads(1, 1, 1)] void main(uint3 dispatchThreadID
                                                   : SV_DispatchThreadID)
{
    // Test array extensions with constraints
    testArrayExtension();

    // Test namespace extensions
    testNamespaceExtensions();

    // Type instantiation tests
    GenericType<half>.State state1;
    GenericType<float>.InnerType.Options options1;
    NonGenericType.Config config1;
    Container.Nested.Settings settings1;
    Container.Nested.DeepNested.Record record1;

    // Initialize extension types with struct literals
    state1 = {1.0h, 2.0h};  // Valid struct initialization
    options1 = {0.0, 1.0};  // Valid struct initialization
    config1 = {0.1, 0.5};   // Valid struct initialization
    settings1 = {0.2, 0.3}; // Valid struct initialization
    record1 = {1, 2.5};     // Valid struct initialization

    // Extension type mismatches - assign value type to extension type
    state1 = 0;    // Error: expected expr of type 'GenericType<half>.State', got 'int'
    options1 = 0;  // Error: expected expr of type 'GenericType<float>.InnerType.Options', got 'int'
    config1 = 0;   // Error: expected expr of type 'NonGenericType.Config', got 'int'
    settings1 = 0; // Error: expected expr of type 'Container.Nested.Settings', got 'int'
    record1 = 0;   // Error: expected expr of type 'Container.Nested.DeepNested.Record', got 'int'

    // Extension type mismatches - assign wrong extension types
    GenericType<int>.InnerType intInner;
    GenericType<float>.InnerType floatInner;

    // This should fail due to different generic parameters
    floatInner.Options floatOpts =
        intInner.getOptions(); // This won't compile as intInner doesn't have getOptions

    // Extension member access - valid cases
    state1.value = 1.0h;       // Valid
    state1.factor = 2.0h;      // Valid
    options1.min = 0.0;        // Valid
    options1.max = 1.0;        // Valid
    config1.threshold = 0.5;   // Valid
    config1.scale = 2.0;       // Valid
    settings1.value = 0.1;     // Valid
    settings1.threshold = 0.2; // Valid
    record1.id = 100;          // Valid
    record1.value = 3.14;      // Valid

    // Extension function calls
    GenericType<half> halfType;
    NonGenericType nonGenType;
    Container.Nested nested;
    Container.Nested.DeepNested deepNested;
    GenericType<float>.InnerType floatInnerType;

    state1 = halfType.createState();        // Valid
    config1 = nonGenType.createConfig();    // Valid
    settings1 = nested.createSettings();    // Valid
    record1 = deepNested.createRecord(42);  // Valid
    options1 = floatInnerType.getOptions(); // Valid

    // Type mismatches - function return values
    options1 =
        halfType
            .createState(); // Error: expected expr of type 'GenericType<float>.InnerType.Options',
                            // got 'GenericType<half>.State'
    state1 = nonGenType.createConfig(); // Error: expected expr of type 'GenericType<half>.State',
                                        // got 'NonGenericType.Config'
    config1 = nested.createSettings();  // Error: expected expr of type 'NonGenericType.Config', got
                                        // 'Container.Nested.Settings'
}

// Define an interface for constraint
interface IFoo
{
    float getValue();
}

// Struct that implements IFoo
struct Bar : IFoo
{
    float value;

    float getValue() { return value; }
}

// Struct that doesn't implement IFoo
struct Baz
{
    float value;
}

// Extension with constrained generic type parameter
extension<T : IFoo, let N : int> Array<T, N>
{
    // Add a struct type inside the extension
    struct DataStats
    {
        float average;
        float maximum;
        float minimum;
    }

    // Add a function that uses the struct type
    DataStats computeStats()
    {
        DataStats stats;
        stats.average = 0;
        stats.maximum = -1e38;
        stats.minimum = 1e38;

        for(int i = 0; i < N; i++)
        {
            float val = this[i].getValue();
            stats.average += val;
            stats.maximum = max(stats.maximum, val);
            stats.minimum = min(stats.minimum, val);
        }

        if(N > 0)
            stats.average /= float(N);

        return stats;
    }

    // Add a field
    static DataStats defaultStats;
}

void testArrayExtension()
{
    // Create an array of a type that implements IFoo
    Bar barArray[3] = { {1.0}, {2.0}, {3.0} };

    // Create an array of a type that doesn't implement IFoo
    Baz bazArray[3] = { {1.0}, {2.0}, {3.0} };

    // This should work - using the extension on a valid type
    Array<Bar, 3>.DataStats barStats;
    barStats = barArray.computeStats(); // Valid

    // Type mismatch errors
    barStats = 0; // Error: expected expr of type 'Bar[3].DataStats', got 'int'

    // This won't compile - Baz doesn't implement IFoo
    //Array<Baz, 3>.DataStats bazStats;  // This should fail because Baz doesn't implement IFoo
    //bazStats = bazArray.computeStats(); // This also won't compile

    // Type mismatch between different extension instantiations
    Array<Bar, 2>.DataStats bar2Stats;
    bar2Stats = barArray.computeStats(); // Error: expected expr of type 'Bar[2].DataStats', got 'Bar[3].DataStats'
}

// Test namespace extensions
// Define a simple base struct
struct SimpleBase
{
    float value;
}

// Define a namespace with an extension
namespace TestNamespace
{
    extension SimpleBase
    {
        struct NamedConfig
        {
            float factor;
        }

        NamedConfig createConfig()
        {
            NamedConfig config;
            config.factor = value * 2.0;
            return config;
        }
    }
}

// Test function for namespace extensions
void testNamespaceExtensions()
{
    // Need to use the namespace to access the extension methods
    using namespace TestNamespace;

    SimpleBase base;
    base.value = 5.0;

    // The type is just "SimpleBase.NamedConfig", not "TestNamespace.SimpleBase.NamedConfig"
    SimpleBase.NamedConfig config;
    config.factor = 2.0;

    // Generate a type error to check the output format
    config = 0; // Error: expected expr of type 'SimpleBase.NamedConfig', got 'int'
}
