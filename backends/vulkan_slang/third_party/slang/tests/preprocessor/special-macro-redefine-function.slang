//DIAGNOSTIC_TEST:SIMPLE:-E

// 
// Check replacing __FILE__
// 

#define __FILE__(x) "Is" x

__FILE__("Anybody")
__FILE__(There)

#undef __FILE__

__FILE__