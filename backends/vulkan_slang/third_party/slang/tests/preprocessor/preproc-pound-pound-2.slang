//DIAGNOSTIC_TEST:SIMPLE:-E

#define A a
#define B b
#define OBJ A ## B

// Should output AB 
// Old Slang outputs ab

OBJ

