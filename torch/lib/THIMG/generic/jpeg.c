#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/jpeg.c"
#else


/******************** JPEG DECOMPRESSION SAMPLE INTERFACE *******************/

/* This half of the example shows how to read data from the JPEG decompressor.
 * It's a bit more refined than the above, in that we show:
 *   (a) how to modify the JPEG library's standard error-reporting behavior;
 *   (b) how to allocate workspace using the library's memory manager.
 *
 * Just to make this example a little different from the first one, we'll
 * assume that we do not intend to put the whole image into an in-memory
 * buffer, but to send it line-by-line someplace else.  We need a one-
 * scanline-high JSAMPLE array as a work buffer, and we will let the JPEG
 * memory manager allocate it for us.  This approach is actually quite useful
 * because we don't need to remember to deallocate the buffer separately: it
 * will go away automatically when the JPEG object is cleaned up.
 */


/*
 * ERROR HANDLING:
 *
 * The JPEG library's standard error handler (jerror.c) is divided into
 * several "methods" which you can override individually.  This lets you
 * adjust the behavior without duplicating a lot of code, which you might
 * have to update with each future release.
 *
 * Our example here shows how to override the "error_exit" method so that
 * control is returned to the library's caller when a fatal error occurs,
 * rather than calling exit() as the standard error_exit method does.
 *
 * We use C's setjmp/longjmp facility to return control.  This means that the
 * routine which calls the JPEG library must first execute a setjmp() call to
 * establish the return point.  We want the replacement error_exit to do a
 * longjmp().  But we need to make the setjmp buffer accessible to the
 * error_exit routine.  To do this, we make a private extension of the
 * standard JPEG error handler object.  (If we were using C++, we'd say we
 * were making a subclass of the regular error handler.)
 *
 * Here's the extended error handler struct:
 */

#ifndef _LIBJPEG_ERROR_STRUCTS_
#define _LIBJPEG_ERROR_STRUCTS_
struct my_error_mgr {
  struct jpeg_error_mgr pub;  /* "public" fields */

  jmp_buf setjmp_buffer;  /* for return to caller */

  char msg[JMSG_LENGTH_MAX]; /* last error message */
};

typedef struct my_error_mgr * my_error_ptr;
#endif

/*
 * Here's the routine that will replace the standard error_exit method:
 */

METHODDEF(void)
THIMG_(JPEG_error) (j_common_ptr cinfo)
{
  /* cinfo->err really points to a my_error_mgr struct, so coerce pointer */
  my_error_ptr myerr = (my_error_ptr) cinfo->err;

  /* See below. */
  (*cinfo->err->output_message) (cinfo);

  /* Return control to the setjmp point */
  longjmp(myerr->setjmp_buffer, 1);
}

/*
 * Here's the routine that will replace the standard output_message method:
 */

METHODDEF(void)
THIMG_(JPEG_output_message) (j_common_ptr cinfo)
{
  my_error_ptr myerr = (my_error_ptr) cinfo->err;

  (*cinfo->err->format_message) (cinfo, myerr->msg);
}


/*
 * Sample routine for JPEG decompression.  We assume that the source file name
 * is passed in.  We want to return 1 on success, 0 on error.
 */

void THIMG_(JPEG_size)(
          const char *filename,
          THLongStorage *result)
{
  if (result == NULL) {
    THError("Argument 'result' must not be NULL.");
  }

  /* This struct contains the JPEG decompression parameters and pointers to
   * working space (which is allocated as needed by the JPEG library).
   */
  struct jpeg_decompress_struct cinfo;
  /* We use our private extension JPEG error handler.
   * Note that this struct must live as long as the main JPEG parameter
   * struct, to avoid dangling-pointer problems.
   */
  struct my_error_mgr jerr;
  /* More stuff */
  FILE * infile;    /* source file */

  /* In this example we want to open the input file before doing anything else,
   * so that the setjmp() error recovery below can assume the file is open.
   * VERY IMPORTANT: use "b" option to fopen() if you are on a machine that
   * requires it in order to read binary files.
   */

  if ((infile = fopen(filename, "rb")) == NULL) {
    THError("cannot open file <%s> for reading", filename);
  }

  /* Step 1: allocate and initialize JPEG decompression object */

  /* We set up the normal JPEG error routines, then override error_exit. */
  cinfo.err = jpeg_std_error(&jerr.pub);
  jerr.pub.error_exit = THIMG_(JPEG_error);
  jerr.pub.output_message = THIMG_(JPEG_output_message);
  /* Establish the setjmp return context for my_error_exit to use. */
  if (setjmp(jerr.setjmp_buffer)) {
    /* If we get here, the JPEG code has signaled an error.
     * We need to clean up the JPEG object, close the input file, and return.
     */
    jpeg_destroy_decompress(&cinfo);
    fclose(infile);
    THError(jerr.msg);
  }

  /* Now we can initialize the JPEG decompression object. */
  jpeg_create_decompress(&cinfo);

  /* Step 2: specify data source (eg, a file) */

  jpeg_stdio_src(&cinfo, infile);

  /* Step 3: read file parameters with jpeg_read_header() */

  jpeg_read_header(&cinfo, TRUE);
  /* We can ignore the return value from jpeg_read_header since
   *   (a) suspension is not possible with the stdio data source, and
   *   (b) we passed TRUE to reject a tables-only JPEG file as an error.
   * See libjpeg.doc for more info.
   */

  /* Step 4: set parameters for decompression */

  /* In this example, we don't need to change any of the defaults set by
   * jpeg_read_header(), so we do nothing here.
   */

  /* Step 5: Start decompressor */

  (void) jpeg_start_decompress(&cinfo);
  /* We can ignore the return value since suspension is not possible
   * with the stdio data source.
   */

  THLongStorage_resize(result, 3);
  THLongStorage_set(result, 0, cinfo.output_components);
  THLongStorage_set(result, 1, cinfo.output_height);
  THLongStorage_set(result, 2, cinfo.output_width);

  /* Step 8: Release JPEG decompression object */

  /* This is an important step since it will release a good deal of memory. */
  jpeg_destroy_decompress(&cinfo);

  /* After finish_decompress, we can close the input file.
   * Here we postpone it until after no more JPEG errors are possible,
   * so as to simplify the setjmp error logic above.  (Actually, I don't
   * think that jpeg_destroy can do an error exit, but why assume anything...)
   */
  fclose(infile);

  /* At this point you may want to check to see whether any corrupt-data
   * warnings occurred (test whether jerr.pub.num_warnings is nonzero).
   */
}


void THIMG_(JPEG_load)(
          int load_from_file,
          const char *filename,
          THByteTensor *src,
          THTensor *result)
{
  if (result == NULL) {
    THError("Argument 'result' must not be NULL.");
  }

#if !defined(HAVE_JPEG_MEM_SRC)
  if (load_from_file != 1) {
    THError(JPEG_MEM_SRC_ERR_MSG);
  }
#endif

  /* This struct contains the JPEG decompression parameters and pointers to
   * working space (which is allocated as needed by the JPEG library).
   */
  struct jpeg_decompress_struct cinfo;
  /* We use our private extension JPEG error handler.
   * Note that this struct must live as long as the main JPEG parameter
   * struct, to avoid dangling-pointer problems.
   */
  struct my_error_mgr jerr;
  /* More stuff */
  FILE * infile;		    /* source file (if loading from file) */
  unsigned char * inmem;    /* source memory (if loading from memory) */
  unsigned long inmem_size; /* source memory size (bytes) */
  JSAMPARRAY buffer;		/* Output row buffer */
  /* int row_stride;		/1* physical row width in output buffer *1/ */
  int i, k;

  if (load_from_file == 1) {
    /* In this example we want to open the input file before doing anything else,
     * so that the setjmp() error recovery below can assume the file is open.
     * VERY IMPORTANT: use "b" option to fopen() if you are on a machine that
     * requires it in order to read binary files.
     */

    if ((infile = fopen(filename, "rb")) == NULL) {
      THError("cannot open file <%s> for reading", filename);
    }
  } else {
    /* We're loading from a ByteTensor */
    if (src == NULL) {
      THError("source ByteTensor must not be NULL");
    }
    inmem = THByteTensor_data(src);
    inmem_size = src->size[0];
    infile = NULL;
  }

  /* Step 1: allocate and initialize JPEG decompression object */

  /* We set up the normal JPEG error routines, then override error_exit. */
  cinfo.err = jpeg_std_error(&jerr.pub);
  jerr.pub.error_exit = THIMG_(JPEG_error);
  jerr.pub.output_message = THIMG_(JPEG_output_message);
  /* Establish the setjmp return context for my_error_exit to use. */
  if (setjmp(jerr.setjmp_buffer)) {
    /* If we get here, the JPEG code has signaled an error.
     * We need to clean up the JPEG object, close the input file, and return.
     */
    jpeg_destroy_decompress(&cinfo);
    if (infile) {
      fclose(infile);
    }
    THError(jerr.msg);
  }
  /* Now we can initialize the JPEG decompression object. */
  jpeg_create_decompress(&cinfo);

  /* Step 2: specify data source (eg, a file) */
  if (load_from_file == 1) {
    jpeg_stdio_src(&cinfo, infile);
  } else {
    jpeg_mem_src(&cinfo, inmem, inmem_size);
  }

  /* Step 3: read file parameters with jpeg_read_header() */

  (void) jpeg_read_header(&cinfo, TRUE);
  /* We can ignore the return value from jpeg_read_header since
   *   (a) suspension is not possible with the stdio data source, and
   *   (b) we passed TRUE to reject a tables-only JPEG file as an error.
   * See libjpeg.doc for more info.
   */

  /* Step 4: set parameters for decompression */

  /* In this example, we don't need to change any of the defaults set by
   * jpeg_read_header(), so we do nothing here.
   */

  /* Step 5: Start decompressor */

  (void) jpeg_start_decompress(&cinfo);
  /* We can ignore the return value since suspension is not possible
   * with the stdio data source.
   */

  /* We may need to do some setup of our own at this point before reading
   * the data.  After jpeg_start_decompress() we have the correct scaled
   * output image dimensions available, as well as the output colormap
   * if we asked for color quantization.
   * In this example, we need to make an output work buffer of the right size.
   */

  /* Make a one-row-high sample array that will go away when done with image */
  const unsigned int chans = cinfo.output_components;
  const unsigned int height = cinfo.output_height;
  const unsigned int width = cinfo.output_width;
  THTensor_(resize3d)(result, chans, height, width);
  real *tdata = THTensor_(data)(result);
  buffer = (*cinfo.mem->alloc_sarray)
    ((j_common_ptr) &cinfo, JPOOL_IMAGE, chans * width, 1);

  /* Step 6: while (scan lines remain to be read) */
  /*           jpeg_read_scanlines(...); */

  /* Here we use the library's state variable cinfo.output_scanline as the
   * loop counter, so that we don't have to keep track ourselves.
   */
  while (cinfo.output_scanline < height) {
    /* jpeg_read_scanlines expects an array of pointers to scanlines.
     * Here the array is only one element long, but you could ask for
     * more than one scanline at a time if that's more convenient.
     */
    (void) jpeg_read_scanlines(&cinfo, buffer, 1);
    const unsigned int j = cinfo.output_scanline-1;

    if (chans == 3) { /* special-case for speed */
      real *td1 = tdata + 0 * (height * width) + j * width;
      real *td2 = tdata + 1 * (height * width) + j * width;
      real *td3 = tdata + 2 * (height * width) + j * width;
      const unsigned char *buf = buffer[0];
      for(i = 0; i < width; i++) {
        *td1++ = (real)buf[chans * i + 0];
        *td2++ = (real)buf[chans * i + 1];
        *td3++ = (real)buf[chans * i + 2];
      }
    } else if (chans == 1) { /* special-case for speed */
      real *td = tdata + j * width;
      for(i = 0; i < width; i++) {
        *td++ = (real)buffer[0][i];
      }
    } else { /* general case */
      for(k = 0; k < chans; k++) {
        const unsigned int k_ = k;
        real *td = tdata + k_ * (height * width) + j * width;
        for(i = 0; i < width; i++) {
          *td++ = (real)buffer[0][chans * i + k_];
        }
      }
    }
  }
  /* Step 7: Finish decompression */

  (void) jpeg_finish_decompress(&cinfo);
  /* We can ignore the return value since suspension is not possible
   * with the stdio data source.
   */

  /* Step 8: Release JPEG decompression object */

  /* This is an important step since it will release a good deal of memory. */
  jpeg_destroy_decompress(&cinfo);

  /* After finish_decompress, we can close the input file.
   * Here we postpone it until after no more JPEG errors are possible,
   * so as to simplify the setjmp error logic above.  (Actually, I don't
   * think that jpeg_destroy can do an error exit, but why assume anything...)
   */
  if (infile) {
    fclose(infile);
  }

  /* At this point you may want to check to see whether any corrupt-data
   * warnings occurred (test whether jerr.pub.num_warnings is nonzero).
   */
}


/*
 * save function
 *
 */
void THIMG_(JPEG_save)(
          const char *filename,
          THTensor *tensor,
          int save_to_file,
          int quality,
          THByteTensor *tensor_dest)
{
#if !defined(HAVE_JPEG_MEM_DEST)
  if (save_to_file != 1) {
    THError(JPEG_MEM_DEST_ERR_MSG);
  }
#endif

  unsigned char *inmem = NULL;  /* destination memory (if saving to memory) */
  unsigned long inmem_size = 0;  /* destination memory size (bytes) */

  THTensor *tensorc = THTensor_(newContiguous)(tensor);
  real *tensor_data = THTensor_(data)(tensorc);

  if (quality < 0 || quality > 100) {
    THError("quality should be between 0 and 100");
  }

  /* jpeg struct */
  struct jpeg_compress_struct cinfo;
  struct jpeg_error_mgr jerr;

  /* pointer to raw image */
  unsigned char *raw_image = NULL;

  /* dimensions of the image we want to write */
  int width=0, height=0, bytes_per_pixel=0;
  int color_space=0;
  if (tensorc->nDimension == 3) {
    bytes_per_pixel = tensorc->size[0];
    height = tensorc->size[1];
    width = tensorc->size[2];
    if (bytes_per_pixel == 3) {
      color_space = JCS_RGB;
    } else if (bytes_per_pixel == 1) {
      color_space = JCS_GRAYSCALE;
    } else {
      THError("tensor should have 1 or 3 channels (gray or RGB)");
    }
  } else if (tensorc->nDimension == 2) {
    bytes_per_pixel = 1;
    height = tensorc->size[0];
    width = tensorc->size[1];
    color_space = JCS_GRAYSCALE;
  } else {
    THError("supports only 1 or 3 dimension tensors");
  }

  /* alloc raw image data */
  raw_image = (unsigned char *)malloc((sizeof (unsigned char))*width*height*bytes_per_pixel);

  /* convert tensor to raw bytes */
  int x,y,k;
  for (k=0; k<bytes_per_pixel; k++) {
    for (y=0; y<height; y++) {
      for (x=0; x<width; x++) {
        raw_image[(y*width+x)*bytes_per_pixel+k] = *tensor_data++;
      }
    }
  }

  /* this is a pointer to one row of image data */
  JSAMPROW row_pointer[1];
  FILE *outfile = NULL;
  if (save_to_file == 1) {
    outfile = fopen( filename, "wb" );
    if (!outfile) {
      THError("Error opening output jpeg file %s\n!", filename);
    }
  }

  cinfo.err = jpeg_std_error( &jerr );
  jpeg_create_compress(&cinfo);

  /* specify data source (eg, a file) */
  if (save_to_file == 1) {
    jpeg_stdio_dest(&cinfo, outfile);
  } else {
    jpeg_mem_dest(&cinfo, &inmem, &inmem_size);
  }

  /* Setting the parameters of the output file here */
  cinfo.image_width = width;
  cinfo.image_height = height;
  cinfo.input_components = bytes_per_pixel;
  cinfo.in_color_space = color_space;

  /* default compression parameters, we shouldn't be worried about these */
  jpeg_set_defaults( &cinfo );
  jpeg_set_quality(&cinfo, quality, (boolean)0);

  /* Now do the compression .. */
  jpeg_start_compress( &cinfo, TRUE );

  /* like reading a file, this time write one row at a time */
  while (cinfo.next_scanline < cinfo.image_height) {
    row_pointer[0] = &raw_image[ cinfo.next_scanline * cinfo.image_width *  cinfo.input_components];
    jpeg_write_scanlines( &cinfo, row_pointer, 1 );
  }

  /* similar to read file, clean up after we're done compressing */
  jpeg_finish_compress( &cinfo );
  jpeg_destroy_compress( &cinfo );

  if (outfile != NULL) {
    fclose( outfile );
  }

  if (save_to_file == 0) {
    THByteTensor_resize1d(tensor_dest, inmem_size);  /* will fail if it's not a Byte Tensor */
    unsigned char* tensor_dest_data = THByteTensor_data(tensor_dest);
    memcpy(tensor_dest_data, inmem, inmem_size);
    free(inmem);
  }

  /* some cleanup */
  free(raw_image);
  THTensor_(free)(tensorc);
}


#endif
