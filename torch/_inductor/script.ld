SECTIONS {
  /* By default, in LLD 16, .lrodata is placed immediately after .rodata.
   * However, .lrodata can be very large in our compiled models, which leads to
   * relocation out-of-range errors for relative relocations. So we place it
   * after other the sections that are referenced from .text using relative
   * relocations. This is the default behavior in GNU ld. */
  .lrodata : { *(.lrodata) }
 } INSERT AFTER .bss;
