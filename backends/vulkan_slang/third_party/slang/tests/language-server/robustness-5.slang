//TEST:LANG_SERVER:
//
//Broken syntax
//HOVER:9,11

__attributeTarget(StructDecl)
attribute_syntax [MyAttribute(name:String)];
[MyAttribute]
struct MyStruct
{
    int a = 5;
    
}