
export void ispcCompilerId() {

// Identify the compiler
#if defined(ISPC)
  print("INFO:compiler[Intel]");
#endif

// Identify the platform
#if defined(__linux) || defined(__linux__) || defined(linux)
  print("INFO:platform[Linux]");
#elif defined(__CYGWIN__)
  print("INFO:platform[Cygwin]");
#elif defined(__MINGW32__)
  print("INFO:platform[MinGW]");
#elif defined(__APPLE__)
  print("INFO:platform[Darwin]");
#elif defined(_WIN32) || defined(__WIN32__) || defined(WIN32)
  print("INFO:platform[Windows]");
#elif defined(__FreeBSD__) || defined(__FreeBSD)
  print("INFO:platform[FreeBSD]");
#elif defined(__NetBSD__) || defined(__NetBSD)
  print("INFO:platform[NetBSD]");
#elif defined(__OpenBSD__) || defined(__OPENBSD)
  print("INFO:platform[OpenBSD]");
#elif defined(__sun) || defined(sun)
  print("INFO:platform[SunOS]");
#elif defined(_AIX) || defined(__AIX) || defined(__AIX__) || defined(__aix) || defined(__aix__)
  print("INFO:platform[AIX]");
#elif defined(__hpux) || defined(__hpux__)
  print("INFO:platform[HP-UX]");
#elif defined(__HAIKU__)
  print("INFO:platform[Haiku]");
#elif defined(__BeOS) || defined(__BEOS__) || defined(_BEOS)
  print("INFO:platform[BeOS]");
#elif defined(__QNX__) || defined(__QNXNTO__)
  print("INFO:platform[QNX]");
#elif defined(__tru64) || defined(_tru64) || defined(__TRU64__)
  print("INFO:platform[Tru64]");
#elif defined(__riscos) || defined(__riscos__)
  print("INFO:platform[RISCos]");
#elif defined(__sinix) || defined(__sinix__) || defined(__SINIX__)
  print("INFO:platform[SINIX]");
#elif defined(__UNIX_SV__)
  print("INFO:platform[UNIX_SV]");
#elif defined(__bsdos__)
  print("INFO:platform[BSDOS]");
#elif defined(_MPRAS) || defined(MPRAS)
  print("INFO:platform[MP-RAS]");
#elif defined(__osf) || defined(__osf__)
  print("INFO:platform[OSF1]");
#elif defined(_SCO_SV) || defined(SCO_SV) || defined(sco_sv)
  print("INFO:platform[SCO_SV]");
#elif defined(__ultrix) || defined(__ultrix__) || defined(_ULTRIX)
  print("INFO:platform[ULTRIX]");
#elif defined(__XENIX__) || defined(_XENIX) || defined(XENIX)
  print("INFO:platform[Xenix]");
#else
  print("INFO:platform[]");
#endif

}
