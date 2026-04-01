//TEST_IGNORE_FILE:

implementing main;

// Cyclic include, OK.
__include a;

int f_a() { return 1; }
