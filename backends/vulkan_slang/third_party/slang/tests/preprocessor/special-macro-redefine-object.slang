//DIAGNOSTIC_TEST:SIMPLE:-E

// 
// Check replacing __LINE__
// 

#define __LINE__ "Hello"

__LINE__

#undef __LINE__

__LINE__
