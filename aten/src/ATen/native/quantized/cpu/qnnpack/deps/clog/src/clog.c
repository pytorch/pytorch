/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif
#ifdef __ANDROID__
#include <android/log.h>
#endif

#ifndef CLOG_LOG_TO_STDIO
#ifdef __ANDROID__
#define CLOG_LOG_TO_STDIO 0
#else
#define CLOG_LOG_TO_STDIO 1
#endif
#endif

#include <clog.h>

/* Messages up to this size are formatted entirely on-stack, and don't allocate
 * heap memory */
#define CLOG_STACK_BUFFER_SIZE 1024

#define CLOG_FATAL_PREFIX "Fatal error: "
#define CLOG_FATAL_PREFIX_LENGTH 13
#define CLOG_FATAL_PREFIX_FORMAT "Fatal error in %s: "
#define CLOG_ERROR_PREFIX "Error: "
#define CLOG_ERROR_PREFIX_LENGTH 7
#define CLOG_ERROR_PREFIX_FORMAT "Error in %s: "
#define CLOG_WARNING_PREFIX "Warning: "
#define CLOG_WARNING_PREFIX_LENGTH 9
#define CLOG_WARNING_PREFIX_FORMAT "Warning in %s: "
#define CLOG_INFO_PREFIX "Note: "
#define CLOG_INFO_PREFIX_LENGTH 6
#define CLOG_INFO_PREFIX_FORMAT "Note (%s): "
#define CLOG_DEBUG_PREFIX "Debug: "
#define CLOG_DEBUG_PREFIX_LENGTH 7
#define CLOG_DEBUG_PREFIX_FORMAT "Debug (%s): "
#define CLOG_SUFFIX_LENGTH 1

void clog_vlog_fatal(const char* module, const char* format, va_list args) {
#if defined(__ANDROID__) && !CLOG_LOG_TO_STDIO
  __android_log_vprint(ANDROID_LOG_FATAL, module, format, args);
#else
  char stack_buffer[CLOG_STACK_BUFFER_SIZE];
  char* heap_buffer = NULL;
  char* out_buffer = &stack_buffer[0];

  /* The first call to vsnprintf will clobber args, thus need a copy in case a
   * second vsnprintf call is needed */
  va_list args_copy;
  va_copy(args_copy, args);

  int prefix_chars = CLOG_FATAL_PREFIX_LENGTH;
  if (module == NULL) {
    memcpy(stack_buffer, CLOG_FATAL_PREFIX, CLOG_FATAL_PREFIX_LENGTH);
  } else {
    prefix_chars = snprintf(
        stack_buffer, CLOG_STACK_BUFFER_SIZE, CLOG_FATAL_PREFIX_FORMAT, module);
    if (prefix_chars < 0) {
      /* Format error in prefix (possible if prefix is modified): skip prefix
       * and continue as if nothing happened. */
      prefix_chars = 0;
    }
  }

  int format_chars;
  if (prefix_chars + CLOG_SUFFIX_LENGTH >= CLOG_STACK_BUFFER_SIZE) {
    /*
     * Prefix + suffix alone would overflow the on-stack buffer, thus need to
     * use on-heap buffer. Do not even try to format the string into on-stack
     * buffer.
     */
    format_chars = vsnprintf(NULL, 0, format, args);
  } else {
    format_chars = vsnprintf(
        &stack_buffer[prefix_chars],
        CLOG_STACK_BUFFER_SIZE - prefix_chars - CLOG_SUFFIX_LENGTH,
        format,
        args);
  }
  if (format_chars < 0) {
    /* Format error in the message: silently ignore this particular message. */
    goto cleanup;
  }
  if (prefix_chars + format_chars + CLOG_SUFFIX_LENGTH >
      CLOG_STACK_BUFFER_SIZE) {
    /* Allocate a buffer on heap, and vsnprintf to this buffer */
    heap_buffer = malloc(prefix_chars + format_chars + CLOG_SUFFIX_LENGTH);
    if (heap_buffer == NULL) {
      goto cleanup;
    }

    if (prefix_chars > CLOG_STACK_BUFFER_SIZE) {
      /* Prefix didn't fit into on-stack buffer, re-format it again to on-heap
       * buffer */
      snprintf(
          heap_buffer,
          prefix_chars + 1 /* for '\0'-terminator */,
          CLOG_FATAL_PREFIX_FORMAT,
          module);
    } else {
      /* Copy pre-formatted prefix from on-stack buffer to on-heap buffer */
      memcpy(heap_buffer, stack_buffer, prefix_chars);
    }
    vsnprintf(
        heap_buffer + prefix_chars,
        format_chars + CLOG_SUFFIX_LENGTH,
        format,
        args_copy);
    out_buffer = heap_buffer;
  }
  out_buffer[prefix_chars + format_chars] = '\n';
#ifdef _WIN32
  DWORD bytes_written;
  WriteFile(
      GetStdHandle(STD_ERROR_HANDLE),
      out_buffer,
      prefix_chars + format_chars + CLOG_SUFFIX_LENGTH,
      &bytes_written,
      NULL);
#else
  write(
      STDERR_FILENO,
      out_buffer,
      prefix_chars + format_chars + CLOG_SUFFIX_LENGTH);
#endif

cleanup:
  free(heap_buffer);
  va_end(args_copy);
#endif
}

void clog_vlog_error(const char* module, const char* format, va_list args) {
#if defined(__ANDROID__) && !CLOG_LOG_TO_STDIO
  __android_log_vprint(ANDROID_LOG_ERROR, module, format, args);
#else
  char stack_buffer[CLOG_STACK_BUFFER_SIZE];
  char* heap_buffer = NULL;
  char* out_buffer = &stack_buffer[0];

  /* The first call to vsnprintf will clobber args, thus need a copy in case a
   * second vsnprintf call is needed */
  va_list args_copy;
  va_copy(args_copy, args);

  int prefix_chars = CLOG_ERROR_PREFIX_LENGTH;
  if (module == NULL) {
    memcpy(stack_buffer, CLOG_ERROR_PREFIX, CLOG_ERROR_PREFIX_LENGTH);
  } else {
    prefix_chars = snprintf(
        stack_buffer, CLOG_STACK_BUFFER_SIZE, CLOG_ERROR_PREFIX_FORMAT, module);
    if (prefix_chars < 0) {
      /* Format error in prefix (possible if prefix is modified): skip prefix
       * and continue as if nothing happened. */
      prefix_chars = 0;
    }
  }

  int format_chars;
  if (prefix_chars + CLOG_SUFFIX_LENGTH >= CLOG_STACK_BUFFER_SIZE) {
    /*
     * Prefix + suffix alone would overflow the on-stack buffer, thus need to
     * use on-heap buffer. Do not even try to format the string into on-stack
     * buffer.
     */
    format_chars = vsnprintf(NULL, 0, format, args);
  } else {
    format_chars = vsnprintf(
        &stack_buffer[prefix_chars],
        CLOG_STACK_BUFFER_SIZE - prefix_chars - CLOG_SUFFIX_LENGTH,
        format,
        args);
  }
  if (format_chars < 0) {
    /* Format error in the message: silently ignore this particular message. */
    goto cleanup;
  }
  if (prefix_chars + format_chars + CLOG_SUFFIX_LENGTH >
      CLOG_STACK_BUFFER_SIZE) {
    /* Allocate a buffer on heap, and vsnprintf to this buffer */
    heap_buffer = malloc(prefix_chars + format_chars + CLOG_SUFFIX_LENGTH);
    if (heap_buffer == NULL) {
      goto cleanup;
    }

    if (prefix_chars > CLOG_STACK_BUFFER_SIZE) {
      /* Prefix didn't fit into on-stack buffer, re-format it again to on-heap
       * buffer */
      snprintf(
          heap_buffer,
          prefix_chars + 1 /* for '\0'-terminator */,
          CLOG_ERROR_PREFIX_FORMAT,
          module);
    } else {
      /* Copy pre-formatted prefix from on-stack buffer to on-heap buffer */
      memcpy(heap_buffer, stack_buffer, prefix_chars);
    }
    vsnprintf(
        heap_buffer + prefix_chars,
        format_chars + CLOG_SUFFIX_LENGTH,
        format,
        args_copy);
    out_buffer = heap_buffer;
  }
  out_buffer[prefix_chars + format_chars] = '\n';
#ifdef _WIN32
  DWORD bytes_written;
  WriteFile(
      GetStdHandle(STD_ERROR_HANDLE),
      out_buffer,
      prefix_chars + format_chars + CLOG_SUFFIX_LENGTH,
      &bytes_written,
      NULL);
#else
  write(
      STDERR_FILENO,
      out_buffer,
      prefix_chars + format_chars + CLOG_SUFFIX_LENGTH);
#endif

cleanup:
  free(heap_buffer);
  va_end(args_copy);
#endif
}

void clog_vlog_warning(const char* module, const char* format, va_list args) {
#if defined(__ANDROID__) && !CLOG_LOG_TO_STDIO
  __android_log_vprint(ANDROID_LOG_WARN, module, format, args);
#else
  char stack_buffer[CLOG_STACK_BUFFER_SIZE];
  char* heap_buffer = NULL;
  char* out_buffer = &stack_buffer[0];

  /* The first call to vsnprintf will clobber args, thus need a copy in case a
   * second vsnprintf call is needed */
  va_list args_copy;
  va_copy(args_copy, args);

  int prefix_chars = CLOG_WARNING_PREFIX_LENGTH;
  if (module == NULL) {
    memcpy(stack_buffer, CLOG_WARNING_PREFIX, CLOG_WARNING_PREFIX_LENGTH);
  } else {
    prefix_chars = snprintf(
        stack_buffer,
        CLOG_STACK_BUFFER_SIZE,
        CLOG_WARNING_PREFIX_FORMAT,
        module);
    if (prefix_chars < 0) {
      /* Format error in prefix (possible if prefix is modified): skip prefix
       * and continue as if nothing happened. */
      prefix_chars = 0;
    }
  }

  int format_chars;
  if (prefix_chars + CLOG_SUFFIX_LENGTH >= CLOG_STACK_BUFFER_SIZE) {
    /*
     * Prefix + suffix alone would overflow the on-stack buffer, thus need to
     * use on-heap buffer. Do not even try to format the string into on-stack
     * buffer.
     */
    format_chars = vsnprintf(NULL, 0, format, args);
  } else {
    format_chars = vsnprintf(
        &stack_buffer[prefix_chars],
        CLOG_STACK_BUFFER_SIZE - prefix_chars - CLOG_SUFFIX_LENGTH,
        format,
        args);
  }
  if (format_chars < 0) {
    /* Format error in the message: silently ignore this particular message. */
    goto cleanup;
  }
  if (prefix_chars + format_chars + CLOG_SUFFIX_LENGTH >
      CLOG_STACK_BUFFER_SIZE) {
    /* Allocate a buffer on heap, and vsnprintf to this buffer */
    heap_buffer = malloc(prefix_chars + format_chars + CLOG_SUFFIX_LENGTH);
    if (heap_buffer == NULL) {
      goto cleanup;
    }

    if (prefix_chars > CLOG_STACK_BUFFER_SIZE) {
      /* Prefix didn't fit into on-stack buffer, re-format it again to on-heap
       * buffer */
      snprintf(
          heap_buffer,
          prefix_chars + 1 /* for '\0'-terminator */,
          CLOG_WARNING_PREFIX_FORMAT,
          module);
    } else {
      /* Copy pre-formatted prefix from on-stack buffer to on-heap buffer */
      memcpy(heap_buffer, stack_buffer, prefix_chars);
    }
    vsnprintf(
        heap_buffer + prefix_chars,
        format_chars + CLOG_SUFFIX_LENGTH,
        format,
        args_copy);
    out_buffer = heap_buffer;
  }
  out_buffer[prefix_chars + format_chars] = '\n';
#ifdef _WIN32
  DWORD bytes_written;
  WriteFile(
      GetStdHandle(STD_ERROR_HANDLE),
      out_buffer,
      prefix_chars + format_chars + CLOG_SUFFIX_LENGTH,
      &bytes_written,
      NULL);
#else
  write(
      STDERR_FILENO,
      out_buffer,
      prefix_chars + format_chars + CLOG_SUFFIX_LENGTH);
#endif

cleanup:
  free(heap_buffer);
  va_end(args_copy);
#endif
}

void clog_vlog_info(const char* module, const char* format, va_list args) {
#if defined(__ANDROID__) && !CLOG_LOG_TO_STDIO
  __android_log_vprint(ANDROID_LOG_INFO, module, format, args);
#else
  char stack_buffer[CLOG_STACK_BUFFER_SIZE];
  char* heap_buffer = NULL;
  char* out_buffer = &stack_buffer[0];

  /* The first call to vsnprintf will clobber args, thus need a copy in case a
   * second vsnprintf call is needed */
  va_list args_copy;
  va_copy(args_copy, args);

  int prefix_chars = CLOG_INFO_PREFIX_LENGTH;
  if (module == NULL) {
    memcpy(stack_buffer, CLOG_INFO_PREFIX, CLOG_INFO_PREFIX_LENGTH);
  } else {
    prefix_chars = snprintf(
        stack_buffer, CLOG_STACK_BUFFER_SIZE, CLOG_INFO_PREFIX_FORMAT, module);
    if (prefix_chars < 0) {
      /* Format error in prefix (possible if prefix is modified): skip prefix
       * and continue as if nothing happened. */
      prefix_chars = 0;
    }
  }

  int format_chars;
  if (prefix_chars + CLOG_SUFFIX_LENGTH >= CLOG_STACK_BUFFER_SIZE) {
    /*
     * Prefix + suffix alone would overflow the on-stack buffer, thus need to
     * use on-heap buffer. Do not even try to format the string into on-stack
     * buffer.
     */
    format_chars = vsnprintf(NULL, 0, format, args);
  } else {
    format_chars = vsnprintf(
        &stack_buffer[prefix_chars],
        CLOG_STACK_BUFFER_SIZE - prefix_chars - CLOG_SUFFIX_LENGTH,
        format,
        args);
  }
  if (format_chars < 0) {
    /* Format error in the message: silently ignore this particular message. */
    goto cleanup;
  }
  if (prefix_chars + format_chars + CLOG_SUFFIX_LENGTH >
      CLOG_STACK_BUFFER_SIZE) {
    /* Allocate a buffer on heap, and vsnprintf to this buffer */
    heap_buffer = malloc(prefix_chars + format_chars + CLOG_SUFFIX_LENGTH);
    if (heap_buffer == NULL) {
      goto cleanup;
    }

    if (prefix_chars > CLOG_STACK_BUFFER_SIZE) {
      /* Prefix didn't fit into on-stack buffer, re-format it again to on-heap
       * buffer */
      snprintf(
          heap_buffer,
          prefix_chars + 1 /* for '\0'-terminator */,
          CLOG_INFO_PREFIX_FORMAT,
          module);
    } else {
      /* Copy pre-formatted prefix from on-stack buffer to on-heap buffer */
      memcpy(heap_buffer, stack_buffer, prefix_chars);
    }
    vsnprintf(
        heap_buffer + prefix_chars,
        format_chars + CLOG_SUFFIX_LENGTH,
        format,
        args_copy);
    out_buffer = heap_buffer;
  }
  out_buffer[prefix_chars + format_chars] = '\n';
#ifdef _WIN32
  DWORD bytes_written;
  WriteFile(
      GetStdHandle(STD_OUTPUT_HANDLE),
      out_buffer,
      prefix_chars + format_chars + CLOG_SUFFIX_LENGTH,
      &bytes_written,
      NULL);
#else
  write(
      STDOUT_FILENO,
      out_buffer,
      prefix_chars + format_chars + CLOG_SUFFIX_LENGTH);
#endif

cleanup:
  free(heap_buffer);
  va_end(args_copy);
#endif
}

void clog_vlog_debug(const char* module, const char* format, va_list args) {
#if defined(__ANDROID__) && !CLOG_LOG_TO_STDIO
  __android_log_vprint(ANDROID_LOG_DEBUG, module, format, args);
#else
  char stack_buffer[CLOG_STACK_BUFFER_SIZE];
  char* heap_buffer = NULL;
  char* out_buffer = &stack_buffer[0];

  /* The first call to vsnprintf will clobber args, thus need a copy in case a
   * second vsnprintf call is needed */
  va_list args_copy;
  va_copy(args_copy, args);

  int prefix_chars = CLOG_DEBUG_PREFIX_LENGTH;
  if (module == NULL) {
    memcpy(stack_buffer, CLOG_DEBUG_PREFIX, CLOG_DEBUG_PREFIX_LENGTH);
  } else {
    prefix_chars = snprintf(
        stack_buffer, CLOG_STACK_BUFFER_SIZE, CLOG_DEBUG_PREFIX_FORMAT, module);
    if (prefix_chars < 0) {
      /* Format error in prefix (possible if prefix is modified): skip prefix
       * and continue as if nothing happened. */
      prefix_chars = 0;
    }
  }

  int format_chars;
  if (prefix_chars + CLOG_SUFFIX_LENGTH >= CLOG_STACK_BUFFER_SIZE) {
    /*
     * Prefix + suffix alone would overflow the on-stack buffer, thus need to
     * use on-heap buffer. Do not even try to format the string into on-stack
     * buffer.
     */
    format_chars = vsnprintf(NULL, 0, format, args);
  } else {
    format_chars = vsnprintf(
        &stack_buffer[prefix_chars],
        CLOG_STACK_BUFFER_SIZE - prefix_chars - CLOG_SUFFIX_LENGTH,
        format,
        args);
  }
  if (format_chars < 0) {
    /* Format error in the message: silently ignore this particular message. */
    goto cleanup;
  }
  if (prefix_chars + format_chars + CLOG_SUFFIX_LENGTH >
      CLOG_STACK_BUFFER_SIZE) {
    /* Allocate a buffer on heap, and vsnprintf to this buffer */
    heap_buffer = malloc(prefix_chars + format_chars + CLOG_SUFFIX_LENGTH);
    if (heap_buffer == NULL) {
      goto cleanup;
    }

    if (prefix_chars > CLOG_STACK_BUFFER_SIZE) {
      /* Prefix didn't fit into on-stack buffer, re-format it again to on-heap
       * buffer */
      snprintf(
          heap_buffer,
          prefix_chars + 1 /* for '\0'-terminator */,
          CLOG_DEBUG_PREFIX_FORMAT,
          module);
    } else {
      /* Copy pre-formatted prefix from on-stack buffer to on-heap buffer */
      memcpy(heap_buffer, stack_buffer, prefix_chars);
    }
    vsnprintf(
        heap_buffer + prefix_chars,
        format_chars + CLOG_SUFFIX_LENGTH,
        format,
        args_copy);
    out_buffer = heap_buffer;
  }
  out_buffer[prefix_chars + format_chars] = '\n';
#ifdef _WIN32
  DWORD bytes_written;
  WriteFile(
      GetStdHandle(STD_OUTPUT_HANDLE),
      out_buffer,
      prefix_chars + format_chars + CLOG_SUFFIX_LENGTH,
      &bytes_written,
      NULL);
#else
  write(
      STDOUT_FILENO,
      out_buffer,
      prefix_chars + format_chars + CLOG_SUFFIX_LENGTH);
#endif

cleanup:
  free(heap_buffer);
  va_end(args_copy);
#endif
}
