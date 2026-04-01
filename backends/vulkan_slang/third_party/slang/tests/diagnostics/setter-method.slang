// setter-method.slang

//DIAGNOSTIC_TEST:SIMPLE:

// Make sure we provide a user a diagnostic if they
// try to declare a setter method without `mutating`
// (even if we don't support `mutating` yet).

struct Sphere
{
    float3 center;
    float radius;

    void setCenter(float3 value)
    {
        center = value;
    }

    void setRadius(float value)
    {
        this.radius = value;
    }
};
