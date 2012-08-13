#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/Tensor.c"
#else

static void torch_Tensor_(c_readTensorStorageSizeStride)(lua_State *L, int index, int allowNone, int allowTensor, int allowStorage, int allowStride,
                                                         THStorage **storage_, long *storageOffset_, THLongStorage **size_, THLongStorage **stride_);

static void torch_Tensor_(c_readSizeStride)(lua_State *L, int index, int allowStride, THLongStorage **size_, THLongStorage **stride_);

static int torch_Tensor_(size)(lua_State *L)
{
  THTensor *tensor = luaT_checkudata(L, 1, torch_Tensor);
  if(lua_isnumber(L,2))
  {
    int dim = luaL_checkint(L, 2)-1;
    luaL_argcheck(L, dim >= 0 && dim < tensor->nDimension, 2, "out of range");
    lua_pushnumber(L, tensor->size[dim]);
  }
  else
  {
    THLongStorage *storage = THLongStorage_newWithSize(tensor->nDimension);
    memmove(storage->data, tensor->size, sizeof(long)*tensor->nDimension);
    luaT_pushudata(L, storage, "torch.LongStorage");
  }
  return 1;
}

static int torch_Tensor_(stride)(lua_State *L)
{
  THTensor *tensor = luaT_checkudata(L, 1, torch_Tensor);
  if(lua_isnumber(L,2))
  {
    int dim = luaL_checkint(L, 2)-1;
    luaL_argcheck(L, dim >= 0 && dim < tensor->nDimension, 2, "out of range");
    lua_pushnumber(L, tensor->stride[dim]);
  }
  else
  {
    THLongStorage *storage = THLongStorage_newWithSize(tensor->nDimension);
    memmove(storage->data, tensor->stride, sizeof(long)*tensor->nDimension);
    luaT_pushudata(L, storage, "torch.LongStorage");
  }
  return 1;
}

static int torch_Tensor_(nDimension)(lua_State *L)
{
  THTensor *tensor = luaT_checkudata(L, 1, torch_Tensor);
  lua_pushnumber(L, tensor->nDimension);
  return 1;
}

static int torch_Tensor_(storage)(lua_State *L)
{
  THTensor *tensor = luaT_checkudata(L, 1, torch_Tensor);
  if(tensor->storage)
  {
    THStorage_(retain)(tensor->storage);
    luaT_pushudata(L, tensor->storage, torch_Storage);
  }
  else
    lua_pushnil(L);

  return 1;
}

static int torch_Tensor_(storageOffset)(lua_State *L)
{
  THTensor *tensor = luaT_checkudata(L, 1, torch_Tensor);
  lua_pushnumber(L, tensor->storageOffset+1);
  return 1;
}

static int torch_Tensor_(new)(lua_State *L)
{
  THTensor *tensor;
  long storageOffset;
  THLongStorage *size, *stride;

  if(lua_type(L, 1) == LUA_TTABLE)
  {
    long i, j;
    THLongStorage *counter;
    long si = 0;
    int dimension = 0;
    int is_finished = 0;

    lua_settop(L, 1);
    size = THLongStorage_new();

    while( (lua_type(L, -1) == LUA_TTABLE) && (lua_objlen(L, -1) > 0) )
    {
      THLongStorage_resize(size, dimension+1);
      size->data[dimension] = lua_objlen(L, -1);
      dimension++;
      lua_rawgeti(L, -1, 1);
    }
    lua_pop(L, 1);

    counter = THLongStorage_newWithSize(size->size);
    THLongStorage_fill(counter, 0);

    tensor = THTensor_(newWithSize)(size, NULL);
    
    if(size->size == 0)
      is_finished = 1;
    
    while(!is_finished)
    {
      if(!lua_istable(L, -1))
      {
        THLongStorage_free(size);
        THLongStorage_free(counter);
        THTensor_(free)(tensor);
        luaL_error(L, "invalid tensor definition");
      }

      if(lua_objlen(L, -1) != size->data[size->size-1])
      {
        THLongStorage_free(size);
        THLongStorage_free(counter);
        THTensor_(free)(tensor);
        luaL_error(L, "invalid tensor sizes");
      }

      for(i = 0; i < size->data[size->size-1]; i++)
      {
        lua_rawgeti(L, -1, i+1);
        if(!lua_isnumber(L, -1))
        {
          THLongStorage_free(size);
          THLongStorage_free(counter);
          THTensor_(free)(tensor);
          luaL_error(L, "invalid element (not a number)");
        }
        THStorage_(set)(THTensor_(storage)(tensor), si++, (real)lua_tonumber(L, -1));
        lua_pop(L, 1);
      }
    
      if(size->size == 1)
        break;

      for(i = size->size-2; i >= 0; i--)
      {
        if(++counter->data[i] == size->data[i])
        {
          if(i == 0)
          {
            is_finished = 1;
            break;
          }
          else
          {
            counter->data[i] = 0;
            lua_pop(L, 1);
          }
        }
        else
        {
          lua_pop(L, 1);
          for(j = i; j < size->size-1; j++)
          {
            if(!lua_istable(L, -1))
            {
              THLongStorage_free(size);
              THLongStorage_free(counter);
              THTensor_(free)(tensor);
              luaL_error(L, "invalid tensor definition");
            }
            if(lua_objlen(L, -1) != size->data[j])
            {
              THLongStorage_free(size);
              THLongStorage_free(counter);
              THTensor_(free)(tensor);
              luaL_error(L, "invalid tensor sizes");
            }
            lua_rawgeti(L, -1, counter->data[j]+1);
          }
          break;
        }
      }
    }

    THLongStorage_free(size);
    THLongStorage_free(counter);
  }
  else
  {
    THStorage *storage;

    torch_Tensor_(c_readTensorStorageSizeStride)(L, 1, 1, 1, 1, 1,
                                                 &storage, &storageOffset, &size, &stride);
    
    tensor = THTensor_(newWithStorage)(storage, storageOffset, size, stride);

    THLongStorage_free(size);
    THLongStorage_free(stride);
  }

  luaT_pushudata(L, tensor, torch_Tensor);
  return 1;
}

static int torch_Tensor_(set)(lua_State *L)
{
  THTensor *self = luaT_checkudata(L, 1, torch_Tensor);
  THStorage *storage;
  long storageOffset;
  THLongStorage *size, *stride;

  torch_Tensor_(c_readTensorStorageSizeStride)(L, 2, 1, 1, 1, 1,
                                               &storage, &storageOffset, &size, &stride);

  THTensor_(setStorage)(self, storage, storageOffset, size, stride);

  THLongStorage_free(size);
  THLongStorage_free(stride);

  lua_settop(L, 1);
  return 1;
}

static int torch_Tensor_(clone)(lua_State *L)
{
  THTensor *self = luaT_checkudata(L, 1, torch_Tensor);
  self = THTensor_(newClone)(self);
  luaT_pushudata(L, self, torch_Tensor);
  return 1;
}

static int torch_Tensor_(contiguous)(lua_State *L)
{
  THTensor *self = luaT_checkudata(L, 1, torch_Tensor);
  self = THTensor_(newContiguous)(self);
  luaT_pushudata(L, self, torch_Tensor);
  return 1;
}

/* Resize */
static int torch_Tensor_(resizeAs)(lua_State *L)
{
  THTensor *tensor = luaT_checkudata(L, 1, torch_Tensor);
  THTensor *src = luaT_checkudata(L, 2, torch_Tensor);
  THTensor_(resizeAs)(tensor, src);
  lua_settop(L, 1);
  return 1;
}

static int torch_Tensor_(resize)(lua_State *L)
{
  THTensor *tensor = luaT_checkudata(L, 1, torch_Tensor);
  THLongStorage *size, *stride;

  torch_Tensor_(c_readSizeStride)(L, 2, 0, &size, &stride);

  THTensor_(resize)(tensor, size, stride);

  THLongStorage_free(size);
  THLongStorage_free(stride);

  lua_settop(L, 1);
  return 1;
}

static int torch_Tensor_(narrow)(lua_State *L)
{
  THTensor *tensor = luaT_checkudata(L, 1, torch_Tensor);
  int dimension = luaL_checkint(L, 2)-1;
  long firstIndex = luaL_checklong(L, 3)-1;
  long size = luaL_checklong(L, 4);

/*  THArgCheck( (dimension >= 0) && (dimension < tensor->nDimension), 2, "out of range");
  THArgCheck( (firstIndex >= 0) && (firstIndex < tensor->size[dimension]), 3, "out of range");
  THArgCheck( (size > 0) && (firstIndex+size <= tensor->size[dimension]), 4, "out of range");
*/
  tensor = THTensor_(newWithTensor)(tensor);
  THTensor_(narrow)(tensor, NULL, dimension, firstIndex, size);
  luaT_pushudata(L, tensor, torch_Tensor);
  return 1;
}

static int torch_Tensor_(sub)(lua_State *L)
{
  THTensor *tensor = luaT_checkudata(L, 1, torch_Tensor);
  long d0s = -1, d0e = -1, d1s = -1, d1e = -1, d2s = -1, d2e = -1, d3s = -1, d3e = -1;

  d0s = luaL_checklong(L, 2)-1;
  d0e = luaL_checklong(L, 3)-1;
  if(d0s < 0)
    d0s += tensor->size[0]+1;
  if(d0e < 0)
    d0e += tensor->size[0]+1;
  luaL_argcheck(L, tensor->nDimension > 0, 2, "invalid dimension");
  luaL_argcheck(L, d0s >= 0 && d0s < tensor->size[0], 2, "out of range");
  luaL_argcheck(L, d0e >= 0 && d0e < tensor->size[0], 3, "out of range");
  luaL_argcheck(L, d0e >= d0s, 3, "end smaller than beginning");

  if(!lua_isnone(L, 4))
  {
    d1s = luaL_checklong(L, 4)-1;
    d1e = luaL_checklong(L, 5)-1;
    if(d1s < 0)
      d1s += tensor->size[1]+1;
    if(d1e < 0)
      d1e += tensor->size[1]+1;
    luaL_argcheck(L, tensor->nDimension > 1, 4, "invalid dimension");
    luaL_argcheck(L, d1s >= 0 && d1s < tensor->size[1], 4, "out of range");
    luaL_argcheck(L, d1e >= 0 && d1e < tensor->size[1], 5, "out of range");    
    luaL_argcheck(L, d1e >= d1s, 5, "end smaller than beginning");

    if(!lua_isnone(L, 6))
    {
      d2s = luaL_checklong(L, 6)-1;
      d2e = luaL_checklong(L, 7)-1;
      if(d2s < 0)
        d2s += tensor->size[2]+1;
      if(d2e < 0)
        d2e += tensor->size[2]+1;
      luaL_argcheck(L, tensor->nDimension > 2, 6, "invalid dimension");
      luaL_argcheck(L, d2s >= 0 && d2s < tensor->size[2], 6, "out of range");
      luaL_argcheck(L, d2e >= 0 && d2e < tensor->size[2], 7, "out of range");    
      luaL_argcheck(L, d2e >= d2s, 7, "end smaller than beginning");

      if(!lua_isnone(L, 8))
      {
        d3s = luaL_checklong(L, 8)-1;
        d3e = luaL_checklong(L, 9)-1;
        if(d3s < 0)
          d3s += tensor->size[3]+1;
        if(d3e < 0)
          d3e += tensor->size[3]+1;
        luaL_argcheck(L, tensor->nDimension > 3, 8, "invalid dimension");
        luaL_argcheck(L, d3s >= 0 && d3s < tensor->size[3], 8, "out of range");
        luaL_argcheck(L, d3e >= 0 && d3e < tensor->size[3], 9, "out of range");    
        luaL_argcheck(L, d3e >= d3s, 9, "end smaller than beginning");
      }
    }
  }

  tensor = THTensor_(newWithTensor)(tensor);
  THTensor_(narrow)(tensor, NULL, 0, d0s, d0e-d0s+1);
  if(d1s >= 0)
    THTensor_(narrow)(tensor, NULL, 1, d1s, d1e-d1s+1);
  if(d2s >= 0)
    THTensor_(narrow)(tensor, NULL, 2, d2s, d2e-d2s+1);
  if(d3s >= 0)
    THTensor_(narrow)(tensor, NULL, 3, d3s, d3e-d3s+1);
  luaT_pushudata(L, tensor, torch_Tensor);
  return 1;
}

static int torch_Tensor_(select)(lua_State *L)
{
  THTensor *tensor = luaT_checkudata(L, 1, torch_Tensor);
  int dimension = luaL_checkint(L, 2)-1;
  long sliceIndex = luaL_checklong(L, 3)-1;

/*   THArgCheck(src->nDimension > 1, 1, "cannot select on a vector");
  THArgCheck((dimension >= 0) && (dimension < src->nDimension), 2, "out of range");
  THArgCheck((sliceIndex >= 0) && (sliceIndex < src->size[dimension]), 3, "out of range");
*/

  if(tensor->nDimension > 1)
  {
    tensor = THTensor_(newWithTensor)(tensor);
    THTensor_(select)(tensor, NULL, dimension, sliceIndex);
    luaT_pushudata(L, tensor, torch_Tensor);
  }
  else
  {
    THArgCheck(tensor->nDimension == 1, 1, "empty Tensor");
    lua_pushnumber(L, THTensor_(get1d)(tensor, sliceIndex));
  }

  return 1;
}


static int torch_Tensor_(transpose)(lua_State *L)
{
  THTensor *tensor = luaT_checkudata(L, 1, torch_Tensor);
  int dimension1 = luaL_checkint(L, 2)-1;
  int dimension2 = luaL_checkint(L, 3)-1;

/*
  THArgCheck( (dimension1 >= 0) && (dimension1 < src->nDimension), 2, "out of range");
  THArgCheck( (dimension2 >= 0) && (dimension2 < src->nDimension), 3, "out of range");
*/

  tensor = THTensor_(newWithTensor)(tensor);
  THTensor_(transpose)(tensor, NULL, dimension1, dimension2);
  luaT_pushudata(L, tensor, torch_Tensor);
  return 1;
}

static int torch_Tensor_(t)(lua_State *L)
{
  THTensor *tensor = luaT_checkudata(L, 1, torch_Tensor);

  luaL_argcheck(L, tensor->nDimension == 2, 1, "Tensor must have 2 dimensions");

  tensor = THTensor_(newWithTensor)(tensor);
  THTensor_(transpose)(tensor, NULL, 0, 1);
  luaT_pushudata(L, tensor, torch_Tensor);
  return 1;
}

static int torch_Tensor_(unfold)(lua_State *L)
{
  THTensor *tensor = luaT_checkudata(L, 1, torch_Tensor);
  int dimension = luaL_checkint(L, 2)-1;
  long size = luaL_checklong(L, 3);
  long step = luaL_checklong(L, 4);

/*
  THArgCheck( (src->nDimension > 0), 1, "cannot unfold an empty tensor");
  THArgCheck(dimension < src->nDimension, 2, "out of range");
  THArgCheck(size <= src->size[dimension], 3, "out of range");
*/

  tensor = THTensor_(newWithTensor)(tensor);
  THTensor_(unfold)(tensor, NULL, dimension, size, step);
  luaT_pushudata(L, tensor, torch_Tensor);
  return 1;
}

/* is contiguous? [a bit like in TnXIterator] */
static int torch_Tensor_(isContiguous)(lua_State *L)
{
  THTensor *tensor = luaT_checkudata(L, 1, torch_Tensor);
  lua_pushboolean(L, THTensor_(isContiguous)(tensor));
  return 1;
}

static int torch_Tensor_(nElement)(lua_State *L)
{
  THTensor *tensor = luaT_checkudata(L, 1, torch_Tensor);
  lua_pushnumber(L, THTensor_(nElement)(tensor));
  return 1;
}

static int torch_Tensor_(copy)(lua_State *L)
{
  THTensor *tensor = luaT_checkudata(L, 1, torch_Tensor);
  void *src;
  if( (src = luaT_toudata(L, 2, torch_Tensor)) )
    THTensor_(copy)(tensor, src);
  else if( (src = luaT_toudata(L, 2, "torch.ByteTensor")) )
    THTensor_(copyByte)(tensor, src);
  else if( (src = luaT_toudata(L, 2, "torch.CharTensor")) )
    THTensor_(copyChar)(tensor, src);
  else if( (src = luaT_toudata(L, 2, "torch.ShortTensor")) )
    THTensor_(copyShort)(tensor, src);
  else if( (src = luaT_toudata(L, 2, "torch.IntTensor")) )
    THTensor_(copyInt)(tensor, src);
  else if( (src = luaT_toudata(L, 2, "torch.LongTensor")) )
    THTensor_(copyLong)(tensor, src);
  else if( (src = luaT_toudata(L, 2, "torch.FloatTensor")) )
    THTensor_(copyFloat)(tensor, src);
  else if( (src = luaT_toudata(L, 2, "torch.DoubleTensor")) )
    THTensor_(copyDouble)(tensor, src);
  else
    luaL_typerror(L, 2, "torch.*Tensor");
  lua_settop(L, 1);
  return 1;
}

static int torch_Tensor_(__newindex__)(lua_State *L)
{
  THTensor *tensor = luaT_checkudata(L, 1, torch_Tensor);
  THLongStorage *idx = NULL;
  THByteTensor *mask;

  if(lua_isnumber(L, 2))
  {
    long index = luaL_checklong(L,2)-1;
    if (index < 0) index = tensor->size[0] + index + 1;
    void *src;
    if (lua_isnumber(L,3)) {
      real value = (real)luaL_checknumber(L,3);
      if (tensor->nDimension == 1) {
        luaL_argcheck(L, index >= 0 && index < tensor->size[0], 2, "out of range");
        THStorage_(set)(tensor->storage, tensor->storageOffset+index*tensor->stride[0], value);
      } else {
        tensor = THTensor_(newWithTensor)(tensor);
        THTensor_(narrow)(tensor, NULL, 0, index, 1);
        THTensor_(fill)(tensor, value);
        THTensor_(free)(tensor);
      }
    } else if( (src = luaT_toudata(L, 3, torch_Tensor)) ) {
      tensor = THTensor_(newWithTensor)(tensor);
      THTensor_(narrow)(tensor, NULL, 0, index, 1);
      THTensor_(copy)(tensor, src);
      THTensor_(free)(tensor);
    } else if( (src = luaT_toudata(L, 3, "torch.ByteTensor")) ) {
      tensor = THTensor_(newWithTensor)(tensor);
      THTensor_(narrow)(tensor, NULL, 0, index, 1);
      THTensor_(copyByte)(tensor, src);
      THTensor_(free)(tensor);
    } else if( (src = luaT_toudata(L, 3, "torch.CharTensor")) ) {
      tensor = THTensor_(newWithTensor)(tensor);
      THTensor_(narrow)(tensor, NULL, 0, index, 1);
      THTensor_(copyChar)(tensor, src);
      THTensor_(free)(tensor);
    } else if( (src = luaT_toudata(L, 3, "torch.ShortTensor")) ) {
      tensor = THTensor_(newWithTensor)(tensor);
      THTensor_(narrow)(tensor, NULL, 0, index, 1);
      THTensor_(copyShort)(tensor, src);
      THTensor_(free)(tensor);
    } else if( (src = luaT_toudata(L, 3, "torch.IntTensor")) ) {
      tensor = THTensor_(newWithTensor)(tensor);
      THTensor_(narrow)(tensor, NULL, 0, index, 1);
      THTensor_(copyInt)(tensor, src);
      THTensor_(free)(tensor);
    } else if( (src = luaT_toudata(L, 3, "torch.LongTensor")) ) {
      tensor = THTensor_(newWithTensor)(tensor);
      THTensor_(narrow)(tensor, NULL, 0, index, 1);
      THTensor_(copyLong)(tensor, src);
      THTensor_(free)(tensor);
    } else if( (src = luaT_toudata(L, 3, "torch.FloatTensor")) ) {
      tensor = THTensor_(newWithTensor)(tensor);
      THTensor_(narrow)(tensor, NULL, 0, index, 1);
      THTensor_(copyFloat)(tensor, src);
      THTensor_(free)(tensor);
    } else if( (src = luaT_toudata(L, 3, "torch.DoubleTensor")) ) {
      tensor = THTensor_(newWithTensor)(tensor);
      THTensor_(narrow)(tensor, NULL, 0, index, 1);
      THTensor_(copyDouble)(tensor, src);
      THTensor_(free)(tensor);
    } else {
      luaL_typerror(L, 3, "torch.*Tensor");
    }
    lua_pushboolean(L, 1);
  }
  else if((idx = luaT_toudata(L, 2, "torch.LongStorage")))
  {
    long index = THTensor_(storageOffset)(tensor);
    real value = (real)luaL_checknumber(L,3);
    int dim;

    luaL_argcheck(L, idx->size == tensor->nDimension, 2, "invalid size");

    for(dim = 0; dim < idx->size; dim++)
    {
      long z = idx->data[dim]-1;
      if (z < 0) z = tensor->size[dim] + z + 1;
      luaL_argcheck(L, (z >= 0) && (z < tensor->size[dim]), 2, "index out of bound");
      index += z*tensor->stride[dim];
    }

    THStorage_(set)(tensor->storage, index, value);
    lua_pushboolean(L, 1);
  }
  else if(lua_istable(L, 2))
  {
    tensor = THTensor_(newWithTensor)(tensor);
    
    int dim;
    int cdim = 0;
    int ndims = tensor->nDimension;
    int done = 0;
    for(dim = 0; dim < ndims; dim++)
    {
      lua_rawgeti(L, 2, dim+1);
      if(lua_isnumber(L, -1))
      {
        long z = lua_tonumber(L, -1)-1;
        lua_pop(L, 1);
        if (z < 0) z = tensor->size[cdim] + z + 1;
        luaL_argcheck(L, (z >= 0) && (z < tensor->size[cdim]), 2, "index out of bound");
        if(tensor->nDimension == 1) {
          done = 1;
          real value = (real)luaL_checknumber(L,3);
          THStorage_(set)(tensor->storage, tensor->storageOffset+z*tensor->stride[0], value);
        } else {
          THTensor_(select)(tensor, NULL, cdim, z);
        }
      } 
      else if (lua_istable(L, -1)) 
      {
        lua_rawgeti(L, -1, 1);
        long start = 0;
        long end = tensor->size[cdim]-1;
        if(lua_isnumber(L, -1)) {
          start = lua_tonumber(L, -1)-1;
          end = start;
        }
        lua_pop(L, 1);
        if (start < 0) start = tensor->size[cdim] + start + 1;
        luaL_argcheck(L, (start >= 0) && (start < tensor->size[cdim]), 2, "start index out of bound");

        lua_rawgeti(L, -1, 2);
        if(lua_isnumber(L, -1)) {
          end = lua_tonumber(L, -1)-1;
        }
        lua_pop(L, 2);
        if (end < 0) end = tensor->size[cdim] + end + 1;
        luaL_argcheck(L, (end >= 0) && (end < tensor->size[cdim]), 2, "end index out of bound");

        luaL_argcheck(L, (end >= start), 2, "end index must be greater or equal to start index");

        THTensor_(narrow)(tensor, NULL, cdim++, start, end-start+1);
      }
      else 
      {
        break;
      }
    }
    if(!done) {
      // doing a copy
      void *src;
      if (lua_isnumber(L,3)) {
        THTensor_(fill)(tensor, lua_tonumber(L,3));
      } else if( (src = luaT_toudata(L, 3, torch_Tensor)) ) {
        THTensor_(copy)(tensor, src);
      } else if( (src = luaT_toudata(L, 3, "torch.ByteTensor")) ) {
        THTensor_(copyByte)(tensor, src);
      } else if( (src = luaT_toudata(L, 3, "torch.CharTensor")) ) {
        THTensor_(copyChar)(tensor, src);
      } else if( (src = luaT_toudata(L, 3, "torch.ShortTensor")) ) {
        THTensor_(copyShort)(tensor, src);
      } else if( (src = luaT_toudata(L, 3, "torch.IntTensor")) ) {
        THTensor_(copyInt)(tensor, src);
      } else if( (src = luaT_toudata(L, 3, "torch.LongTensor")) ) {
        THTensor_(copyLong)(tensor, src);
      } else if( (src = luaT_toudata(L, 3, "torch.FloatTensor")) ) {
        THTensor_(copyFloat)(tensor, src);
      } else if( (src = luaT_toudata(L, 3, "torch.DoubleTensor")) ) {
        THTensor_(copyDouble)(tensor, src);
      } else {
        luaL_typerror(L, 3, "torch.*Tensor");
      }
    }
    THTensor_(free)(tensor);
    lua_pushboolean(L, 1);
  }
  else if((mask = luaT_toudata(L, 2, "torch.ByteTensor")))
  {
    THTensor *vals;
    if (lua_isnumber(L, 3))
    {
      THTensor_(maskedFill)(tensor, mask, (real)(luaL_checknumber(L,3)));
    }
    else if((vals = luaT_toudata(L, 3, torch_Tensor)))
    {
      THTensor_(maskedCopy)(tensor, mask, vals);
    }
    else
    {
      luaL_error(L,"number or tensor expected");
    }
  }
  else
    lua_pushboolean(L, 0);

  return 1;
}

static int torch_Tensor_(__index__)(lua_State *L)
{
  THTensor *tensor = luaT_checkudata(L, 1, torch_Tensor);
  THLongStorage *idx = NULL;
  THByteTensor *mask;

  if(lua_isnumber(L, 2))
  {
    long index = luaL_checklong(L,2)-1;
    if (index < 0) index = tensor->size[0] + index + 1;

    luaL_argcheck(L, tensor->nDimension > 0, 1, "empty tensor");
    luaL_argcheck(L, index >= 0 && index < tensor->size[0], 2, "out of range");

    if(tensor->nDimension == 1)
    {
      lua_pushnumber(L, THStorage_(get)(tensor->storage, tensor->storageOffset+index*tensor->stride[0]));
    }
    else
    {
      tensor = THTensor_(newWithTensor)(tensor);
      THTensor_(select)(tensor, NULL, 0, index);
      luaT_pushudata(L, tensor, torch_Tensor);
    }
    lua_pushboolean(L, 1);
    return 2;
  }
  else if((idx = luaT_toudata(L, 2, "torch.LongStorage")))
  {
    long index = THTensor_(storageOffset)(tensor);
    int dim;

    luaL_argcheck(L, idx->size == tensor->nDimension, 2, "invalid size");
    
    for(dim = 0; dim < idx->size; dim++)
    {
      long z = idx->data[dim]-1;
      if (z < 0) z = tensor->size[dim] + z + 1;
      luaL_argcheck(L, (z >= 0) && (z < tensor->size[dim]), 2, "index out of bound");
      index += z*tensor->stride[dim];
    }
    lua_pushnumber(L, (double)THStorage_(get)(THTensor_(storage)(tensor), index));
    lua_pushboolean(L, 1);
    return 2;
  }
  else if(lua_istable(L, 2))
  {
    tensor = THTensor_(newWithTensor)(tensor);
    
    int dim;
    int cdim = 0;
    int ndims = tensor->nDimension;
    int done = 0;
    for(dim = 0; dim < ndims; dim++)
    {
      lua_rawgeti(L, 2, dim+1);
      if(lua_isnumber(L, -1)) 
      {
        long z = lua_tonumber(L, -1)-1;
        lua_pop(L, 1);
        if (z < 0) z = tensor->size[cdim] + z + 1;
        luaL_argcheck(L, (z >= 0) && (z < tensor->size[cdim]), 2, "index out of bound");
        if(tensor->nDimension == 1) {
          done = 1;
          lua_pushnumber(L, THStorage_(get)(tensor->storage, tensor->storageOffset+z*tensor->stride[0]));
        } else {
          THTensor_(select)(tensor, NULL, cdim, z);
        }
      } 
      else if (lua_istable(L, -1))
      {
        lua_rawgeti(L, -1, 1);
        long start = 0;
        long end = tensor->size[cdim]-1;
        if(lua_isnumber(L, -1)) {
          start = lua_tonumber(L, -1)-1;
          end = start;
        }
        lua_pop(L, 1);
        if (start < 0) start = tensor->size[cdim] + start + 1;
        luaL_argcheck(L, (start >= 0) && (start < tensor->size[cdim]), 2, "start index out of bound");

        lua_rawgeti(L, -1, 2);
        if(lua_isnumber(L, -1)) {
          end = lua_tonumber(L, -1)-1;
        }
        lua_pop(L, 2);
        if (end < 0) end = tensor->size[cdim] + end + 1;
        luaL_argcheck(L, (end >= 0) && (end < tensor->size[cdim]), 2, "end index out of bound");

        luaL_argcheck(L, (end >= start), 2, "end index must be greater or equal to start index");

        THTensor_(narrow)(tensor, NULL, cdim++, start, end-start+1);
      }
      else 
      {
        break;
      }
    }
    if(!done) {
      luaT_pushudata(L, tensor, torch_Tensor);
    } else {
      THTensor_(free)(tensor);
    }
    lua_pushboolean(L, 1);
    return 2;
  }
  else if((mask = luaT_toudata(L, 2, "torch.ByteTensor")))
  {
    THTensor *vals = THTensor_(new)();
    THTensor_(maskedSelect)(vals, tensor, mask);
    luaT_pushudata(L, vals, torch_Tensor);
    lua_pushboolean(L, 1);
    return 2;
  }
  else
  {
    lua_pushboolean(L, 0);
    return 1;
  }
}

static int torch_Tensor_(free)(lua_State *L)
{
  THTensor *tensor = luaT_checkudata(L, 1, torch_Tensor);
  THTensor_(free)(tensor);
  return 0;
}

/* helpful functions */
static void torch_Tensor_(c_readSizeStride)(lua_State *L, int index, int allowStride, THLongStorage **size_, THLongStorage **stride_)
{
  THLongStorage *size = NULL;
  THLongStorage *stride = NULL;
  
  if( (size = luaT_toudata(L, index, "torch.LongStorage")) )
  {
    if(!lua_isnoneornil(L, index+1))
    {
      if( (stride = luaT_toudata(L, index+1, "torch.LongStorage")) )
        luaL_argcheck(L, stride->size == size->size, index+1, "provided stride and size are inconsistent");
      else
        luaL_argcheck(L, 0, index+1, "torch.LongStorage expected");
    }
    THLongStorage_retain(size);
    if(stride)
      THLongStorage_retain(stride);
  }
  else
  {
    int i;

    size = THLongStorage_newWithSize(8);
    stride = THLongStorage_newWithSize(8);
    THLongStorage_fill(size, -1);
    THLongStorage_fill(stride, -1);

    if(allowStride)
    {
      for(i = 0; i < 8; i++)
      {
        if(lua_isnone(L, index+2*i))
          break;
        size->data[i] = luaL_checklong(L, index+2*i);
        
        if(lua_isnone(L, index+2*i+1))
          break;
        stride->data[i] = luaL_checklong(L, index+2*i+1);
      }
    }
    else
    {
      for(i = 0; i < 8; i++)
      {
        if(lua_isnone(L, index+i))
          break;
        size->data[i] = luaL_checklong(L, index+i);
      }
    }
  }

  *size_ = size;
  *stride_ = stride;
}

static void torch_Tensor_(c_readTensorStorageSizeStride)(lua_State *L, int index, int allowNone, int allowTensor, int allowStorage, int allowStride,
                                                         THStorage **storage_, long *storageOffset_, THLongStorage **size_, THLongStorage **stride_)
{
  static char errMsg[64];
  THTensor *src = NULL;
  THStorage *storage = NULL;

  int arg1Type = lua_type(L, index);

  if( allowNone && (arg1Type == LUA_TNONE) )
  {
    *storage_ = NULL;
    *storageOffset_ = 0;
    *size_ = NULL;
    *stride_ = NULL;
    return;
  }
  else if( allowTensor && (arg1Type == LUA_TUSERDATA) && (src = luaT_toudata(L, index, torch_Tensor)) )
  {
    *storage_ = src->storage;
    *storageOffset_ = src->storageOffset;
    *size_ = THTensor_(newSizeOf)(src);
    *stride_ = THTensor_(newStrideOf)(src);
    return;
  }
  else if( allowStorage && (arg1Type == LUA_TUSERDATA) && (storage = luaT_toudata(L, index, torch_Storage)) )
  {
    *storage_ = storage;
    if(lua_isnone(L, index+1))
    {
      *storageOffset_ = 0;
      *size_ = THLongStorage_newWithSize1(storage->size);
      *stride_ = THLongStorage_newWithSize1(1);
    }
    else
    {
      *storageOffset_ = luaL_checklong(L, index+1)-1;
      torch_Tensor_(c_readSizeStride)(L, index+2, allowStride, size_, stride_);
    }
    return;
  }
  else if( (arg1Type == LUA_TNUMBER) || (luaT_toudata(L, index, "torch.LongStorage")) )
  {
    *storage_ = NULL;
    *storageOffset_ = 0;
    torch_Tensor_(c_readSizeStride)(L, index, 0, size_, stride_);

    return;
  }

  *storage_ = NULL;
  *storageOffset_ = 0;

  sprintf(errMsg, "expecting number%s%s", (allowTensor ? " or Tensor" : ""), (allowStorage ? " or Storage" : ""));
  luaL_argcheck(L, 0, index, errMsg);
}

static int torch_Tensor_(apply)(lua_State *L)
{
  THTensor *tensor = luaT_checkudata(L, 1, torch_Tensor);
  luaL_checktype(L, 2, LUA_TFUNCTION);
  lua_settop(L, 2);

  TH_TENSOR_APPLY(real, tensor,
                  lua_pushvalue(L, 2);
                  lua_pushnumber(L, *tensor_data);
                  lua_call(L, 1, 1);
                  if(lua_isnumber(L, 3))
                  {
                    *tensor_data = (real)lua_tonumber(L, 3);
                    lua_pop(L, 1);
                  }
                  else if(lua_isnil(L, 3))
                    lua_pop(L, 1);
                  else
                    luaL_error(L, "given function should return a number or nil"););

  lua_settop(L, 1);
  return 1;
}

static int torch_Tensor_(map)(lua_State *L)
{
  THTensor *tensor = luaT_checkudata(L, 1, torch_Tensor);
  THTensor *src = luaT_checkudata(L, 2, torch_Tensor);
  luaL_checktype(L, 3, LUA_TFUNCTION);
  lua_settop(L, 3);

  TH_TENSOR_APPLY2(real, tensor, real, src,
                  lua_pushvalue(L, 3);
                  lua_pushnumber(L, *tensor_data);
                  lua_pushnumber(L, *src_data);
                  lua_call(L, 2, 1);
                  if(lua_isnumber(L, 4))
                  {
                    *tensor_data = (real)lua_tonumber(L, 4);
                    lua_pop(L, 1);
                  }
                  else if(lua_isnil(L, 4))
                    lua_pop(L, 1);
                  else
                    luaL_error(L, "given function should return a number or nil"););

  lua_settop(L, 1);
  return 1;
}

static int torch_Tensor_(map2)(lua_State *L)
{
  THTensor *tensor = luaT_checkudata(L, 1, torch_Tensor);
  THTensor *src1 = luaT_checkudata(L, 2, torch_Tensor);
  THTensor *src2 = luaT_checkudata(L, 3, torch_Tensor);
  luaL_checktype(L, 4, LUA_TFUNCTION);
  lua_settop(L, 4);

  TH_TENSOR_APPLY3(real, tensor, real, src1, real, src2,
                  lua_pushvalue(L, 4);
                  lua_pushnumber(L, *tensor_data);
                  lua_pushnumber(L, *src1_data);
                  lua_pushnumber(L, *src2_data);
                  lua_call(L, 3, 1);
                  if(lua_isnumber(L, 5))
                  {
                    *tensor_data = (real)lua_tonumber(L, 5);
                    lua_pop(L, 1);
                  }
                  else if(lua_isnil(L, 5))
                    lua_pop(L, 1);
                  else
                    luaL_error(L, "given function should return a number or nothing"););

  lua_settop(L, 1);
  return 1;
}

static int torch_Tensor_(factory)(lua_State *L)
{
  THTensor *tensor = THTensor_(new)();
  luaT_pushudata(L, tensor, torch_Tensor);
  return 1;
}

static int torch_Tensor_(write)(lua_State *L)
{
  THTensor *tensor = luaT_checkudata(L, 1, torch_Tensor);
  THFile *file = luaT_checkudata(L, 2, "torch.File");

  THFile_writeIntScalar(file, tensor->nDimension);
  THFile_writeLongRaw(file, tensor->size, tensor->nDimension);
  THFile_writeLongRaw(file, tensor->stride, tensor->nDimension);
  THFile_writeLongScalar(file, tensor->storageOffset+1); /* to respect Lua convention */

  lua_getfield(L, 2, "writeObject"); /* the method */
  lua_pushvalue(L, 2); /* the file */
  /* the storage */
  if(tensor->storage)
  {
    THStorage_(retain)(tensor->storage);
    luaT_pushudata(L, tensor->storage, torch_Storage);
  }
  else
    lua_pushnil(L);

  lua_call(L, 2, 0); /* call the method */

  return 0;
}

static int torch_Tensor_(read)(lua_State *L)
{
  THTensor *tensor = luaT_checkudata(L, 1, torch_Tensor);
  THFile *file = luaT_checkudata(L, 2, "torch.File");

  tensor->nDimension = THFile_readIntScalar(file);
  tensor->size = THAlloc(sizeof(long)*tensor->nDimension);
  tensor->stride = THAlloc(sizeof(long)*tensor->nDimension);
  THFile_readLongRaw(file, tensor->size, tensor->nDimension);
  THFile_readLongRaw(file, tensor->stride, tensor->nDimension);
  tensor->storageOffset = THFile_readLongScalar(file);
  tensor->storageOffset--;  /* to respect Lua convention */

  lua_getfield(L, 2, "readObject"); /* the method */
  lua_pushvalue(L, 2); /* the file */
  lua_call(L, 1, 1); /* call the method */

  tensor->storage = luaT_toudata(L, -1, torch_Storage);
  if(tensor->storage)
    THStorage_(retain)(tensor->storage);

  return 0;
}

static const struct luaL_Reg torch_Tensor_(_) [] = {
  {"contiguous", torch_Tensor_(contiguous)},
  {"size", torch_Tensor_(size)},
  {"__len__", torch_Tensor_(size)},
  {"stride", torch_Tensor_(stride)},
  {"dim", torch_Tensor_(nDimension)},
  {"nDimension", torch_Tensor_(nDimension)},
  {"set", torch_Tensor_(set)},
  {"storage", torch_Tensor_(storage)},
  {"storageOffset", torch_Tensor_(storageOffset)},
  {"clone", torch_Tensor_(clone)},
  {"contiguous", torch_Tensor_(contiguous)},
  {"resizeAs", torch_Tensor_(resizeAs)},
  {"resize", torch_Tensor_(resize)},
  {"narrow", torch_Tensor_(narrow)},
  {"sub", torch_Tensor_(sub)},
  {"select", torch_Tensor_(select)},
  {"transpose", torch_Tensor_(transpose)},
  {"t", torch_Tensor_(t)},
  {"unfold", torch_Tensor_(unfold)},
  {"isContiguous", torch_Tensor_(isContiguous)},
  {"nElement", torch_Tensor_(nElement)},
  {"copy", torch_Tensor_(copy)},
  {"apply", torch_Tensor_(apply)},
  {"map", torch_Tensor_(map)},
  {"map2", torch_Tensor_(map2)},
  {"read", torch_Tensor_(read)},
  {"write", torch_Tensor_(write)},
  {"__index__", torch_Tensor_(__index__)},
  {"__newindex__", torch_Tensor_(__newindex__)},
  {NULL, NULL}
};

void torch_Tensor_(init)(lua_State *L)
{
  luaT_newmetatable(L, torch_Tensor, NULL,
                    torch_Tensor_(new), torch_Tensor_(free), torch_Tensor_(factory));
  luaL_register(L, NULL, torch_Tensor_(_));
  lua_pop(L, 1);
}

#endif
