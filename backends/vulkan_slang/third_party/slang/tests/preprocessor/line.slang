//TEST:SIMPLE:
// #line support

FooA a() { return 0; }

#line 99 "b.slang"
FooB b() { return 0; }

#line default
FooC c() { return 0; }

#line 603 "d.slang"
FooD d() { return 0; }

#line 40
FooE e() { return 0; }

#line
FooF f() { return 0; }
