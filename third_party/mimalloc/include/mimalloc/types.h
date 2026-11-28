# We used to abandon huge pages in order to eagerly deallocate it if freed from another thread.
// Unfortunately, that makes it not possible to visit them during a heap walk or include them in a
// `mi_heap_destroy`. We therefore instead reset/decommit the huge blocks nowadays if freed from
// another thread so the memory becomes "virtually" available (and eventually gets properly freed by
// the owning thread).
#define MI_HUGE_PAGE_ABANDON 1
