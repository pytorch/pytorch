#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/png.c"
#else


/*
 * Copyright 2002-2010 Guillaume Cottenceau.
 *
 * This software may be freely redistributed under the terms
 * of the X11 license.
 *
 * Clement: modified for Torch7.
 */
void THIMG_(PNG_load)(
          int load_from_file,
          const char *file_name,
          THByteTensor *src,
          THTensor *result,
          THIntTensor *result_bit_depth)
{
  png_byte header[8];    // 8 is the maximum size that can be checked

  int width, height, bit_depth;
  png_byte color_type;

  png_structp png_ptr;
  png_infop info_ptr;
  png_bytep * row_pointers;
  size_t fread_ret;
  FILE* fp;
  libpng_inmem_buffer inmem = {0};    /* source memory (if loading from memory) */
  libpng_errmsg errmsg;

  if (load_from_file == 1){
   /* open file and test for it being a png */
    fp = fopen(file_name, "rb");
    if (!fp)
      THError("[read_png_file] File %s could not be opened for reading", file_name);
    fread_ret = fread(header, 1, 8, fp);
    if (fread_ret != 8)
      THError("[read_png_file] File %s error reading header", file_name);
    if (png_sig_cmp(header, 0, 8))
      THError("[read_png_file] File %s is not recognized as a PNG file", file_name);
  } else {
    /* We're loading from a ByteTensor */
    if (src == NULL) {
      THError("source ByteTensor must not be NULL");
    }
    inmem.buffer = THByteTensor_data(src);
    inmem.length = src->size[0];
    inmem.offset = 8;
    fp = NULL;
    if (png_sig_cmp(inmem.buffer, 0, 8))
      THError("[read_png_byte_tensor] ByteTensor is not recognized as a PNG file");
  }
  /* initialize stuff */
  png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);

  if (!png_ptr)
    THError("[read_png] png_create_read_struct failed");

  png_set_error_fn(png_ptr, &errmsg, libpng_error_fn, NULL);

  info_ptr = png_create_info_struct(png_ptr);
  if (!info_ptr) {
    png_destroy_read_struct(&png_ptr, NULL, NULL);
    if (fp) {
      fclose(fp);
    }
    THError("[read_png] png_create_info_struct failed");
  }

  if (setjmp(png_jmpbuf(png_ptr))) {
    png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
    if (fp) {
      fclose(fp);
    }
    THError("[read_png] Error during init_io: %s", errmsg.str);
  }

  if (load_from_file == 1) {
    png_init_io(png_ptr, fp);
  } else {
    /* set the read callback */
    png_set_read_fn(png_ptr,(png_voidp)&inmem, libpng_userReadData);
  }
  png_set_sig_bytes(png_ptr, 8);
  png_read_info(png_ptr, info_ptr);

  width      = png_get_image_width(png_ptr, info_ptr);
  height     = png_get_image_height(png_ptr, info_ptr);
  color_type = png_get_color_type(png_ptr, info_ptr);
  bit_depth  = png_get_bit_depth(png_ptr, info_ptr);

  /* get depth */
  int depth = 0;
  if (color_type == PNG_COLOR_TYPE_RGBA) {
    depth = 4;
  } else if (color_type == PNG_COLOR_TYPE_RGB) {
    depth = 3;
  } else if (color_type == PNG_COLOR_TYPE_GRAY) {
    if (bit_depth < 8) {
      png_set_expand_gray_1_2_4_to_8(png_ptr);
    }
    depth = 1;
  } else if (color_type == PNG_COLOR_TYPE_GA) {
    depth = 2;
  } else if (color_type == PNG_COLOR_TYPE_PALETTE) {
    depth = 3;
    png_set_expand(png_ptr);
  } else {
    png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
    if (fp) {
      fclose(fp);
    }
    THError("[read_png_file] Unknown color space");
  }

  if (bit_depth < 8) {
    png_set_strip_16(png_ptr);
  }

  png_read_update_info(png_ptr, info_ptr);

  /* read file */
  if (setjmp(png_jmpbuf(png_ptr))) {
    png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
    if (fp) {
      fclose(fp);
    }
    THError("[read_png_file] Error during read_image: %s", errmsg.str);
  }

  /* alloc tensor */
  THTensor_(resize3d)(result, depth, height, width);
  real *tensor_data = THTensor_(data)(result);

  /* alloc data in lib format */
  row_pointers = (png_bytep*) malloc(sizeof(png_bytep) * height);
  int y;
  for (y=0; y<height; y++)
    row_pointers[y] = (png_byte*) malloc(png_get_rowbytes(png_ptr,info_ptr));

  /* read image in */
  png_read_image(png_ptr, row_pointers);

  /* convert image to dest tensor */
  int x, k;
  if ((bit_depth == 16) && (sizeof(real) > 1)) {
    for (k=0; k<depth; k++) {
      for (y=0; y<height; y++) {
        png_byte* row = row_pointers[y];
        for (x=0; x<width; x++) {
          // PNG is big-endian
          int val = ((int)row[(x*depth+k)*2] << 8) + row[(x*depth+k)*2+1];
          *tensor_data++ = (real)val;
        }
      }
    }
  } else {
    int stride = 1;
    if (bit_depth == 16) {
      /* PNG has 16 bit color depth, but the tensor type is byte. */
      stride = 2;
    }
    for (k=0; k<depth; k++) {
      for (y=0; y<height; y++) {
        png_byte* row = row_pointers[y];
        for (x=0; x<width; x++) {
          *tensor_data++ = (real)row[(x*depth+k)*stride];
          //png_byte val = row[x*depth+k];
          //THTensor_(set3d)(tensor, k, y, x, (real)val);
        }
      }
    }
  }

  /* cleanup heap allocation */
  for (y=0; y<height; y++)
    free(row_pointers[y]);
  free(row_pointers);

  /* cleanup png structs */
  png_read_end(png_ptr, NULL);
  png_destroy_read_struct(&png_ptr, &info_ptr, NULL);

  /* done with file */
  if (fp) {
    fclose(fp);
  }

  if (bit_depth < 8) {
    bit_depth = 8;
  }

  if (result_bit_depth != NULL) {
    THIntTensor_resize1d(result_bit_depth, 1);
    THIntTensor_set1d(result_bit_depth, 0, bit_depth);
  }
}


void THIMG_(PNG_save)(
          const char *file_name,
          THTensor *tensor)
{
  int width=0, height=0;
  png_byte color_type = 0;
  png_byte bit_depth = 8;

  png_structp png_ptr;
  png_infop info_ptr;
  png_bytep * row_pointers;
  libpng_errmsg errmsg;

  /* get dims and contiguous tensor */
  THTensor *tensorc = THTensor_(newContiguous)(tensor);
  real *tensor_data = THTensor_(data)(tensorc);
  long depth=0;
  if (tensorc->nDimension == 3) {
    depth = tensorc->size[0];
    height = tensorc->size[1];
    width = tensorc->size[2];
  } else if (tensorc->nDimension == 2) {
    depth = 1;
    height = tensorc->size[0];
    width = tensorc->size[1];
  }

  /* depth check */
  if ((depth != 1) && (depth != 3) && (depth != 4)) {
    THError("[write_png_file] Depth must be 1, 3 or 4");
  }
  if (depth == 4) color_type = PNG_COLOR_TYPE_RGBA;
  else if (depth == 3) color_type = PNG_COLOR_TYPE_RGB;
  else if (depth == 1) color_type = PNG_COLOR_TYPE_GRAY;

  /* create file */
  FILE *fp = fopen(file_name, "wb");
  if (!fp)
    THError("[write_png_file] File %s could not be opened for writing", file_name);

  /* initialize stuff */
  png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);

  if (!png_ptr)
    THError("[write_png_file] png_create_write_struct failed");

  png_set_error_fn(png_ptr, &errmsg, libpng_error_fn, NULL);

  info_ptr = png_create_info_struct(png_ptr);
  if (!info_ptr)
    THError("[write_png_file] png_create_info_struct failed");

  if (setjmp(png_jmpbuf(png_ptr)))
    THError("[write_png_file] Error during init_io: %s", errmsg.str);

  png_init_io(png_ptr, fp);

  /* write header */
  if (setjmp(png_jmpbuf(png_ptr)))
    THError("[write_png_file] Error during writing header: %s", errmsg.str);

  png_set_IHDR(png_ptr, info_ptr, width, height,
         bit_depth, color_type, PNG_INTERLACE_NONE,
         PNG_COMPRESSION_TYPE_BASE, PNG_FILTER_TYPE_BASE);

  png_write_info(png_ptr, info_ptr);

  /* convert tensor to 8bit bytes */
  row_pointers = (png_bytep*) malloc(sizeof(png_bytep) * height);
  int y;
  for (y=0; y<height; y++)
    row_pointers[y] = (png_byte*) malloc(png_get_rowbytes(png_ptr,info_ptr));

  /* convert image to dest tensor */
  int x,k;
  for (k=0; k<depth; k++) {
    for (y=0; y<height; y++) {
      png_byte* row = row_pointers[y];
      for (x=0; x<width; x++) {
        //row[x*depth+k] = (png_byte)THTensor_(get3d)(tensor, k, y, x);
        row[x*depth+k] = *tensor_data++;
      }
    }
  }

  /* write bytes */
  if (setjmp(png_jmpbuf(png_ptr)))
    THError("[write_png_file] Error during writing bytes: %s", errmsg.str);

  png_write_image(png_ptr, row_pointers);

  /* end write */
  if (setjmp(png_jmpbuf(png_ptr)))
    THError("[write_png_file] Error during end of write: %s", errmsg.str);

  /* cleanup png structs */
  png_write_end(png_ptr, NULL);
  png_destroy_write_struct(&png_ptr, &info_ptr);

  /* cleanup heap allocation */
  for (y=0; y<height; y++)
    free(row_pointers[y]);
  free(row_pointers);

  /* cleanup */
  fclose(fp);
  THTensor_(free)(tensorc);
}


void THIMG_(PNG_size)(
          const char *filename,
          THLongStorage *result)
{
  if (result == NULL) {
    THError("Argument 'result' must not be NULL.");
  }

  png_byte header[8];    // 8 is the maximum size that can be checked

  int width, height;
  png_byte color_type;

  png_structp png_ptr;
  png_infop info_ptr;
  libpng_errmsg errmsg;
  size_t fread_ret;
  /* open file and test for it being a png */
  FILE *fp = fopen(filename, "rb");
  if (!fp)
    THError("[get_png_size] File %s could not be opened for reading", filename);
  fread_ret = fread(header, 1, 8, fp);
  if (fread_ret != 8)
    THError("[get_png_size] File %s error reading header", filename);

  if (png_sig_cmp(header, 0, 8))
    THError("[get_png_size] File %s is not recognized as a PNG file", filename);

  /* initialize stuff */
  png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);

  if (!png_ptr)
    THError("[get_png_size] png_create_read_struct failed");

  png_set_error_fn(png_ptr, &errmsg, libpng_error_fn, NULL);

  info_ptr = png_create_info_struct(png_ptr);
  if (!info_ptr)
    THError("[get_png_size] png_create_info_struct failed");

  if (setjmp(png_jmpbuf(png_ptr)))
    THError("[get_png_size] Error during init_io: %s", errmsg.str);

  png_init_io(png_ptr, fp);
  png_set_sig_bytes(png_ptr, 8);

  png_read_info(png_ptr, info_ptr);

  width      = png_get_image_width(png_ptr, info_ptr);
  height     = png_get_image_height(png_ptr, info_ptr);
  color_type = png_get_color_type(png_ptr, info_ptr);
  png_read_update_info(png_ptr, info_ptr);

  /* get depth */
  int depth = 0;
  if (color_type == PNG_COLOR_TYPE_RGBA)
    depth = 4;
  else if (color_type == PNG_COLOR_TYPE_RGB)
    depth = 3;
  else if (color_type == PNG_COLOR_TYPE_GRAY)
    depth = 1;
  else if (color_type == PNG_COLOR_TYPE_GA)
    depth = 2;
  else if (color_type == PNG_COLOR_TYPE_PALETTE)
    THError("[get_png_size] unsupported type: PALETTE");
  else
    THError("[get_png_size] Unknown color space");

  /* done with file */
  fclose(fp);

  THLongStorage_resize(result, 3);
  THLongStorage_set(result, 0, depth);
  THLongStorage_set(result, 1, height);
  THLongStorage_set(result, 2, width);
}

#endif
