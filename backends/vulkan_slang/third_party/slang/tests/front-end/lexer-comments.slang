//TEST:SIMPLE:
// confirming that the lexer handles comments correctly

// line comment

/* block comment
*/

/* block comments don't nest
   /*
*/

float f(float f) { return f; }