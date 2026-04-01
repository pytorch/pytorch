Casting in the Slang Compiler
=============================

The following discussion is about casting within the C++ implementation of the slang compiler. 

C++'s built in mechanisms for casting (principally dynamic_cast) is problematic within the slang compiler codebase. Code using 'dynamic_cast' requires RTTI information is available, and that a type that uses it must have a vtbl (have at least one virtual member). Some problems with this...

* There are types which we want to 'dynamic_cast' that do not have, and we do not want to have a Vtbl (for example Slang::IRInst). 
* There are types which a 'dynamic_cast' doesn't do quite what we want (for example casting on Type* derived types typically wants to work on their canonical type)
* We may want to replace use of dynamic_cast in the future for speed/space or other reasons
* It is common in the code base when using a 'smart pointer' type to cast it, but still return a smart pointer 

To deal with these issues we need casting within Slang to follow it's own methodology. In summary it is as follows...

* Use 'as' free function to do a typical 'dynamic like' cast. 
    * 'as' doesn't guarantee the returned pointer points to the same object.
    * For example with Type* it *actually* does the cast on the canonical type which is often a different object. 
* If you want to *literally* do a dynamic cast use 'dynamicCast' free function. 
    * This guarantees the returned pointer points to the same object (like normal dynamic_cast)
* If you want to return a smart pointer from a cast from a smart pointer use the .as or .dynamicCast *methods*
* If you want to determine if an 'as' cast is possible on a smart pointer use the .is method
    * Doing so will produce more efficient code because a new smart pointer does not need to be constructed

These functions will also work with types that do not have Vtbl - like IRInst derived types. 

Both 'as' and 'dynamicCast' handle the case if the pointer is a nullptr, by returning a nullptr. If the cast succeeds the cast pointer is returned otherwise nullptr is returned. If a cast is performed with a free function it always returns a raw pointer. 

So why have 'as' and 'dynamicCast' - they seem sort of similar? The primary difference is dynamicCast *must* always return a pointer to the same object, whilst 'as' *can* return a pointer to a different object if that is the desired 'normal' casting behavior for the type. This is the case for Type* when using 'as' it may return a different object - the 'canonical type' for the Type*. For a concrete example take 'NamedExpressionType', its canonical type is the type the name relates to. If you use 'as' on it - it will produce a pointer to a different object, an object that will not be castable back into a NamedExpressionType.

Also keep in mind that 'as' behavior is based on the pointer type being cast from. For any pointer to a type derived from Type it will cast the canonical type. **BUT** if the pointer is pointing to a Type derived *object*, but the pointer type is *not* derived from Type (like say RefObject*), then 'as' will behave like dynamicCast. 

All this being said 'as' in usage is seen as the 'default' way to do a 'dynamic like' cast with these special behaviour appropriate for the type when necessary.

By having the free function and method versions of 'as' and 'dynamicCast', you can choose if you want a 'raw' or 'smart' pointer type returned from the cast. If you just want to test if something is a certain type, then using as/dynamicCast free functions is the faster way to do it. If you *know* that a raw pointer is ok, because the object will remain in scope, then again using the free function is better because it does less work. But as the examples following show, care is needed because if you get it wrong the object might go out of scope and leave the raw pointer pointing to a deleted object. When in doubt the safe choice is to typically use .as (or .dynamicCast if appropriate) methods. 

Following example shows the different types of casting...

```C++

void someFunction(Decl* decl, Type* type)
{
    RefPtr<Decl> declRefPtr(decl);
    RefPtr<Type> typeRefPtr(type);

    // Use of as
    {
        // Casting with as on a free function returns a raw pointer
        GenericDecl* genericDeclRaw0 = as<GenericDecl>(decl);
        // Free function again returns a raw pointer
        GenericDecl* genericDeclRaw1 = as<GenericDecl>(declRefPtr);

        // Using the as *method* returns a smart pointer holding the cast result
        RefPtr<GenericDecl> genericDeclRefPtr0 = declRefPtr.as<GenericDecl>();
        
        // Of course you can use auto with either
        auto genericDeclRefPtr1 = declRefPtr.as<GenericDecl>();
        
        auto genericDeclRaw2 = as<GenericDecl>(declRefPtr);
    }
    
    // Currently using as on anything not cast *from* Type is the same as dynamicCast.
    // But on Type* sometimes you may want to control the cast
    {
        // With a NamedExpressionType sometimes you don't want 'as' behaviour - if we want to see the information about the name (not the thing 
        // it relates to (the canonical type)
        NamedExpressionType* namedExpressionRawPtr = dynamicCast<NamedExpressionType>(type);
        
        
        // Returns the smart pointer 
        auto namedExpressionRefPtr = typeRefPtr.as<NamedExpressionType>();
    }
    
```

It is important to be aware of what style of cast you use where. Take for example the following function ...
```C++
    RefPtr<Expr> substitute(RefPtr<Expr> expr) const
    {
        return DeclRefBase::Substitute(expr);
    }
``` 
    
If you want to do a cast on it, you need to be careful especially about scope, for example...

```C++
    RefPtr<Expr> expr = ...;
    
    {
        // Whoops! This is a problem. When using the free function, the cast is to a *raw* pointer, so obj 
        // receives a raw pointer. When the RefPtr returned from Substitute goes out of scope (when the statement is left)
        // the ref will be removed and if the ref count was 1 destroyed. Now obj points to a freed object and so a crash is
        // likely to follow in the future! 
        
        auto obj = as<RefObject>(substitute(expr));
    }
    // So how do we avoid this? Well it depends what the function is returning and the scope. If it's returning a smart pointer, 
    // you could use the .as method
    {
        // This can only compile if it is a smart pointer (raw pointers don't have an as method)
        auto obj = substitute(expr).as<RefObject>();
    }

    // Another option is to put the created thing in a smart pointer so you know it's in scope
    {
        RefPtr<Expr> sub = substitute(expr);
        // Ok as long as sub is in scope
        auto obj = as<RefObject>(sub);
       
    }
 
    // More awkwardly you could use free function, but assign to a smart pointer, thus maintaining scope
    {
        RefPtr<RefObject> obj = as<RefObject>(substitute(expr));
    }

```

The following code shows the change in behavior of 'as' is based on the source *pointer* type **NOT** the *object* type..

```C++
    // Derives from Type
    NamedExpressionType* exprType = ...;

    
    // Will be the Type* of the *canonical* type, because the pointer is Type derived and we are using as!
    Type* type0 = as<Type>(exprType);
    // It' going to be pointing to a different object, because type0 is the cast of the *canonical* type, because exprType derives from Type
    SLANG_ASSERT(type0 != exprType);
    
    // If I do a dynamicCast the result is either nullptr or a pointer that *must* point to the same object
    Type* type1 = dynamicCast<Type>(exprType);
    SLANG_ASSERT(type1 == exprType);
    
    
    // Here, the pointer is pointing to a NamedExpressionType derived object. Which derives from Type. BUT our pointer here does *not* derive from type.
    RefObject* refObj = exprType;
    
    // 'as' just looks at the from type, and it doesn't derive from Type (it's just RefObject), so it does regular as, which is dynamicCast
    Type* type2 = as<Type>(refObject);
    
    SLANG_ASSERT(type2 == exprType);
    
    // Finally... 
    
    // Is true even though exprType is a NamedExpression, because the cast is on the canonical type
    SLANG_ASSERT(as<NamedExpression>(exprType) == nullptr);
    
    // dynamicCast is always the same object returned, so must match
    SLANG_ASSERT(dynamicCast<NamedExpression>(exprType) == exprType);
```
