/*
    Copyright (c) 2005-2018 Intel Corporation

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.




*/

/*
    The original source for this example is
    Copyright (c) 1994-2008 John E. Stone
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions
    are met:
    1. Redistributions of source code must retain the above copyright
       notice, this list of conditions and the following disclaimer.
    2. Redistributions in binary form must reproduce the above copyright
       notice, this list of conditions and the following disclaimer in the
       documentation and/or other materials provided with the distribution.
    3. The name of the author may not be used to endorse or promote products
       derived from this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS
    OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
    WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
    ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY
    DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
    DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
    OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
    HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
    LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
    OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
    SUCH DAMAGE.
*/

/*
 * parse.cpp - an UltraLame (tm) parser for simple data files...
 */

// Try preventing lots of GCC warnings about ignored results of fscanf etc.
#if !__INTEL_COMPILER

#if __GNUC__<4 || __GNUC__==4 && __GNUC_MINOR__<5
// For older versions of GCC, disable use of __wur in GLIBC
#undef _FORTIFY_SOURCE
#define _FORTIFY_SOURCE 0
#else
// Starting from 4.5, GCC has a suppression option
#pragma GCC diagnostic ignored "-Wunused-result"
#endif

#endif //__INTEL_COMPILER

#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <ctype.h> /* needed for toupper(), macro.. */

#include "types.h"
#include "api.h"      /* rendering API */

#define PARSE_INTERNAL
#include "parse.h" /* self protos */
#undef PARSE_INTERNAL

static texentry textable[NUMTEXS]; /* texture lookup table */
static texentry defaulttex;     /* The default texture when a lookup fails */
static int numtextures;         /* number of TEXDEF textures               */
static int numobjectsparsed;    /* total number of objects parsed so far   */
static color scenebackcol;   /* scene background color                  */

static int stringcmp(const char * a, const char * b) {
  size_t i, s, l;

  s=strlen(a);
  l=strlen(b);

  if (s != l)
    return 1;

  for (i=0; i<s; i++) {
    if (toupper(a[i]) != toupper(b[i])) {
      return 1;
    }
  }
  return 0;
}

static void reset_tex_table(void) {
  apitexture apitex;

  numtextures=0;
  memset(&textable, 0, sizeof(textable));

  apitex.col.r=1.0;
  apitex.col.g=1.0;
  apitex.col.b=1.0;
  apitex.ambient=0.1;
  apitex.diffuse=0.9;
  apitex.specular=0.0;
  apitex.opacity=1.0;
  apitex.texturefunc=0;

  defaulttex.tex=rt_texture(&apitex);
}

static errcode add_texture(void * tex, char name[TEXNAMELEN]) {
  textable[numtextures].tex=tex;
  strcpy(textable[numtextures].name, name);

  numtextures++;
  if (numtextures > NUMTEXS) {
    fprintf(stderr, "Parse: %d textures allocated, texture slots full!\n", numtextures);
    numtextures--; /* keep writing over last texture if we've run out.. */
    return PARSEALLOCERR;
  }

  return PARSENOERR;
}

static void * find_texture(char name[TEXNAMELEN]) {
  int i;

  for (i=0; i<numtextures; i++) {
    if (strcmp(name, textable[i].name) == 0)
    return textable[i].tex;
  }
  fprintf(stderr, "Undefined texture '%s', using default. \n",name);
  return(defaulttex.tex);
}

apiflt degtorad(apiflt deg) {
  apiflt tmp;
  tmp=deg * 3.1415926 / 180.0;
  return tmp;
}

static void degvectoradvec(vector * degvec) {
  vector tmp;

  tmp.x=degtorad(degvec->x);
  tmp.y=degtorad(degvec->y);
  tmp.z=degtorad(degvec->z);
  *degvec=tmp;
}

static void InitRot3d(RotMat * rot, apiflt x, apiflt y, apiflt z) {
  rot->rx1=cos(y)*cos(z);
  rot->rx2=sin(x)*sin(y)*cos(z) - cos(x)*sin(z);
  rot->rx3=sin(x)*sin(z) + cos(x)*cos(z)*sin(y);

  rot->ry1=cos(y)*sin(z);
  rot->ry2=cos(x)*cos(z) + sin(x)*sin(y)*sin(z);
  rot->ry3=cos(x)*sin(y)*sin(z) - sin(x)*cos(z);

  rot->rz1=sin(y);
  rot->rz2=sin(x)*cos(y);
  rot->rz3=cos(x)*cos(y);
}

static void Rotate3d(RotMat * rot, vector * vec) {
  vector tmp;
  tmp.x=(vec->x*(rot->rx1) + vec->y*(rot->rx2) + vec->z*(rot->rx3));
  tmp.y=(vec->x*(rot->ry1) + vec->y*(rot->ry2) + vec->z*(rot->ry3));
  tmp.z=(vec->x*(rot->rz1) + vec->y*(rot->rz2) + vec->z*(rot->rz3));
  *vec=tmp;
}

static void Scale3d(vector * scale, vector * vec) {
  vec->x=vec->x * scale->x;
  vec->y=vec->y * scale->y;
  vec->z=vec->z * scale->z;
}

static void Trans3d(vector * trans, vector * vec) {
  vec->x+=trans->x;
  vec->y+=trans->y;
  vec->z+=trans->z;
}

static errcode GetString(FILE * dfile, const char * string) {
  char data[255];

  fscanf(dfile,"%s",data);
  if (stringcmp(data, string) != 0) {
    fprintf(stderr, "parse: Expected %s, got %s \n",string, data);
    fprintf(stderr, "parse: Error while parsing object: %d \n",numobjectsparsed);
    return PARSEBADSYNTAX;
  }

  return PARSENOERR;
}

unsigned int readmodel(char * modelfile, SceneHandle scene) {
  FILE * dfile;
  errcode rc;

  reset_tex_table();
  dfile=NULL;

  dfile=fopen(modelfile,"r");
  if (dfile==NULL) {
    return PARSEBADFILE;
  }

  rc = GetScenedefs(dfile, scene);
  if (rc != PARSENOERR) {
    fclose(dfile);
    return rc;
  }

  scenebackcol.r = 0.0; /* default background is black */
  scenebackcol.g = 0.0;
  scenebackcol.b = 0.0;

  numobjectsparsed=0;
  while ((rc = GetObject(dfile, scene)) == PARSENOERR) {
    numobjectsparsed++;
  }
  fclose(dfile);

  if (rc == PARSEEOF)
    rc = PARSENOERR;

  rt_background(scene, scenebackcol);

  return rc;
}


static errcode GetScenedefs(FILE * dfile, SceneHandle scene) {
  vector Ccenter, Cview, Cup;
  apiflt zoom, aspectratio;
  int raydepth, antialiasing;
  char outfilename[200];
  int xres, yres, verbose;
  float a,b,c;
  errcode rc = PARSENOERR;

  rc |= GetString(dfile, "BEGIN_SCENE");

  rc |= GetString(dfile, "OUTFILE");
  fscanf(dfile, "%s", outfilename);
#ifdef _WIN32
  if (strcmp (outfilename, "/dev/null") == 0) {
    strcpy (outfilename, "NUL:");
  }
#endif

  rc |= GetString(dfile, "RESOLUTION");
  fscanf(dfile, "%d %d", &xres, &yres);

  rc |= GetString(dfile, "VERBOSE");
  fscanf(dfile, "%d", &verbose);

  rt_scenesetup(scene, outfilename, xres, yres, verbose);

  rc |= GetString(dfile, "CAMERA");

  rc |= GetString(dfile, "ZOOM");
  fscanf(dfile, "%f", &a);
  zoom=a;

  rc |= GetString(dfile, "ASPECTRATIO");
  fscanf(dfile, "%f", &b);
  aspectratio=b;

  rc |= GetString(dfile, "ANTIALIASING");
  fscanf(dfile, "%d", &antialiasing);

  rc |= GetString(dfile, "RAYDEPTH");
  fscanf(dfile, "%d", &raydepth);

  rc |= GetString(dfile, "CENTER");
  fscanf(dfile,"%f %f %f", &a, &b, &c);
  Ccenter.x = a;
  Ccenter.y = b;
  Ccenter.z = c;

  rc |= GetString(dfile, "VIEWDIR");
  fscanf(dfile,"%f %f %f", &a, &b, &c);
  Cview.x = a;
  Cview.y = b;
  Cview.z = c;

  rc |= GetString(dfile, "UPDIR");
  fscanf(dfile,"%f %f %f", &a, &b, &c);
  Cup.x = a;
  Cup.y = b;
  Cup.z = c;

  rc |= GetString(dfile, "END_CAMERA");   

  rt_camerasetup(scene, zoom, aspectratio, antialiasing, raydepth,
              Ccenter, Cview, Cup);


  return rc;
}

static errcode GetObject(FILE * dfile, SceneHandle scene) {
  char objtype[80];

  fscanf(dfile, "%s", objtype);
  if (!stringcmp(objtype, "END_SCENE")) {
    return PARSEEOF; /* end parsing */
  }
  if (!stringcmp(objtype, "TEXDEF")) {
    return GetTexDef(dfile);
  }
  if (!stringcmp(objtype, "TEXALIAS")) {
    return GetTexAlias(dfile);
  }
  if (!stringcmp(objtype, "BACKGROUND")) {
    return GetBackGnd(dfile);
  }
  if (!stringcmp(objtype, "CYLINDER")) {
    return GetCylinder(dfile);
  }
  if (!stringcmp(objtype, "FCYLINDER")) {
    return GetFCylinder(dfile);
  }
  if (!stringcmp(objtype, "POLYCYLINDER")) {
    return GetPolyCylinder(dfile);
  }
  if (!stringcmp(objtype, "SPHERE")) {
    return GetSphere(dfile);
  }
  if (!stringcmp(objtype, "PLANE")) {
    return GetPlane(dfile);
  }
  if (!stringcmp(objtype, "RING")) {
    return GetRing(dfile);
  }
  if (!stringcmp(objtype, "BOX")) {
    return GetBox(dfile);
  }
  if (!stringcmp(objtype, "SCALARVOL")) {
    return GetVol(dfile);
  }
  if (!stringcmp(objtype, "TRI")) {
    return GetTri(dfile);
  }
  if (!stringcmp(objtype, "STRI")) {
    return GetSTri(dfile);
  }
  if (!stringcmp(objtype, "LIGHT")) {
    return GetLight(dfile);
  }
  if (!stringcmp(objtype, "SCAPE")) {
    return GetLandScape(dfile);
  }
  if (!stringcmp(objtype, "TPOLYFILE")) {
    return GetTPolyFile(dfile);
  }

  fprintf(stderr, "Found bad token: %s expected an object type\n", objtype);
  return PARSEBADSYNTAX;
}

static errcode GetVector(FILE * dfile, vector * v1) {
  float a, b, c;

  fscanf(dfile, "%f %f %f", &a, &b, &c);
  v1->x=a;
  v1->y=b;
  v1->z=c;

  return PARSENOERR;
}

static errcode GetColor(FILE * dfile, color * c1) {
  float r, g, b;
  int rc;

  rc = GetString(dfile, "COLOR");
  fscanf(dfile, "%f %f %f", &r, &g, &b);
  c1->r=r;
  c1->g=g;
  c1->b=b;

  return rc;
}

static errcode GetTexDef(FILE * dfile) {
  char texname[TEXNAMELEN];

  fscanf(dfile, "%s", texname);
  add_texture(GetTexBody(dfile), texname);

  return PARSENOERR;
}

static errcode GetTexAlias(FILE * dfile) {
  char texname[TEXNAMELEN];
  char aliasname[TEXNAMELEN];

  fscanf(dfile, "%s", texname);
  fscanf(dfile, "%s", aliasname);
  add_texture(find_texture(aliasname), texname);

  return PARSENOERR;
}


static errcode GetTexture(FILE * dfile, void ** tex) {
  char tmp[255];
  errcode rc = PARSENOERR;

  fscanf(dfile, "%s", tmp);
  if (!stringcmp("TEXTURE", tmp)) {
    *tex = GetTexBody(dfile);
  }
  else
    *tex = find_texture(tmp);

  return rc;
}

void * GetTexBody(FILE * dfile) {
  char tmp[255];
  float a,b,c,d, phong, phongexp, phongtype;
  apitexture tex;
  void * voidtex;
  errcode rc;

  rc = GetString(dfile, "AMBIENT");
  fscanf(dfile, "%f", &a);
  tex.ambient=a;

  rc |= GetString(dfile, "DIFFUSE");
  fscanf(dfile, "%f", &b);
  tex.diffuse=b;

  rc |= GetString(dfile, "SPECULAR");
  fscanf(dfile, "%f", &c);
  tex.specular=c;

  rc |= GetString(dfile, "OPACITY");
  fscanf(dfile, "%f", &d);
  tex.opacity=d;

  fscanf(dfile, "%s", tmp);
  if (!stringcmp("PHONG", tmp)) {
    fscanf(dfile, "%s", tmp);
    if (!stringcmp("METAL", tmp)) {
      phongtype = RT_PHONG_METAL;
    }
    else if (!stringcmp("PLASTIC", tmp)) {
      phongtype = RT_PHONG_PLASTIC;
    }
    else {
      phongtype = RT_PHONG_PLASTIC;
    }

    fscanf(dfile, "%f", &phong);
    GetString(dfile, "PHONG_SIZE");
    fscanf(dfile, "%f", &phongexp);
    fscanf(dfile, "%s", tmp);
  }
  else {
    phong = 0.0;
    phongexp = 100.0;
    phongtype = RT_PHONG_PLASTIC;
  }

  fscanf(dfile, "%f %f %f", &a, &b, &c);
  tex.col.r = a;
  tex.col.g = b;
  tex.col.b = c;

  rc |= GetString(dfile, "TEXFUNC");
  fscanf(dfile, "%d", &tex.texturefunc);
  if (tex.texturefunc >= 7) {    /* if its an image map, we need a filename */
    fscanf(dfile, "%s", tex.imap);
  }
  if (tex.texturefunc != 0) {
    rc |= GetString(dfile, "CENTER");
    rc |= GetVector(dfile, &tex.ctr);
    rc |= GetString(dfile, "ROTATE");
    rc |= GetVector(dfile, &tex.rot);
    rc |= GetString(dfile, "SCALE");
    rc |= GetVector(dfile, &tex.scale);
  }
  if (tex.texturefunc == 9) {
    rc |= GetString(dfile, "UAXIS");
    rc |= GetVector(dfile, &tex.uaxs);
    rc |= GetString(dfile, "VAXIS");
    rc |= GetVector(dfile, &tex.vaxs);
  }

  voidtex = rt_texture(&tex);
  rt_tex_phong(voidtex, phong, phongexp, (int) phongtype);

  return voidtex;
}

static errcode GetLight(FILE * dfile) {
  apiflt rad;
  vector ctr;
  apitexture tex;
  float a;
  errcode rc;

  memset(&tex, 0, sizeof(apitexture));

  rc = GetString(dfile,"CENTER");
  rc |= GetVector(dfile, &ctr);
  rc |= GetString(dfile,"RAD");
  fscanf(dfile,"%f",&a);  /* read in radius */
  rad=a;

  rc |= GetColor(dfile, &tex.col);

  rt_light(rt_texture(&tex), ctr, rad);

  return rc;
}

static errcode GetBackGnd(FILE * dfile) {
  float r,g,b;

  fscanf(dfile, "%f %f %f", &r, &g, &b);

  scenebackcol.r=r;
  scenebackcol.g=g;
  scenebackcol.b=b;

  return PARSENOERR;
}

static errcode GetCylinder(FILE * dfile) {
  apiflt rad;
  vector ctr, axis;
  void * tex;
  float a;
  errcode rc;

  rc = GetString(dfile, "CENTER");
  rc |= GetVector(dfile, &ctr);
  rc |= GetString(dfile, "AXIS");
  rc |= GetVector(dfile, &axis);
  rc |= GetString(dfile, "RAD");
  fscanf(dfile, "%f", &a);
  rad=a;

  rc |= GetTexture(dfile, &tex);
  rt_cylinder(tex, ctr, axis, rad);

  return rc;
}

static errcode GetFCylinder(FILE * dfile) {
  apiflt rad;
  vector ctr, axis;
  vector pnt1, pnt2;
  void * tex;
  float a;
  errcode rc;

  rc = GetString(dfile, "BASE");
  rc |= GetVector(dfile, &pnt1);
  rc |= GetString(dfile, "APEX");
  rc |= GetVector(dfile, &pnt2);

  ctr=pnt1;
  axis.x=pnt2.x - pnt1.x;
  axis.y=pnt2.y - pnt1.y;
  axis.z=pnt2.z - pnt1.z;

  rc |= GetString(dfile, "RAD");
  fscanf(dfile, "%f", &a);
  rad=a;

  rc |= GetTexture(dfile, &tex);
  rt_fcylinder(tex, ctr, axis, rad);

  return rc;
}

static errcode GetPolyCylinder(FILE * dfile) {
  apiflt rad;
  vector * temp;
  void * tex;
  float a;
  int numpts, i;
  errcode rc;

  rc = GetString(dfile, "POINTS");
  fscanf(dfile, "%d", &numpts);

  temp = (vector *) malloc(numpts * sizeof(vector));

  for (i=0; i<numpts; i++) {
    rc |= GetVector(dfile, &temp[i]);
  }

  rc |= GetString(dfile, "RAD");
  fscanf(dfile, "%f", &a);
  rad=a;

  rc |= GetTexture(dfile, &tex);
  rt_polycylinder(tex, temp, numpts, rad);

  free(temp);

  return rc;
}


static errcode GetSphere(FILE * dfile) {
  apiflt rad;
  vector ctr;
  void * tex;
  float a;
  errcode rc;

  rc = GetString(dfile,"CENTER");
  rc |= GetVector(dfile, &ctr);
  rc |= GetString(dfile, "RAD");
  fscanf(dfile,"%f",&a);
  rad=a;

  rc |= GetTexture(dfile, &tex);

  rt_sphere(tex, ctr, rad);

  return rc;
}

static errcode GetPlane(FILE * dfile) {
  vector normal;
  vector ctr;
  void * tex;
  errcode rc;

  rc = GetString(dfile, "CENTER");
  rc |= GetVector(dfile, &ctr);
  rc |= GetString(dfile, "NORMAL");
  rc |= GetVector(dfile, &normal);
  rc |= GetTexture(dfile, &tex);

  rt_plane(tex, ctr, normal);

  return rc;
}

static errcode GetVol(FILE * dfile) {
  vector min, max;
  int x,y,z;
  char fname[255];
  void * tex;
  errcode rc;

  rc = GetString(dfile, "MIN");
  rc |= GetVector(dfile, &min);
  rc |= GetString(dfile, "MAX");
  rc |= GetVector(dfile, &max);
  rc |= GetString(dfile, "DIM");
  fscanf(dfile, "%d %d %d ", &x, &y, &z);
  rc |= GetString(dfile, "FILE");
  fscanf(dfile, "%s", fname);
  rc |= GetTexture(dfile, &tex);

  rt_scalarvol(tex, min, max, x, y, z, fname, NULL);

  return rc;
}

static errcode GetBox(FILE * dfile) {
  vector min, max;
  void * tex;
  errcode rc;

  rc = GetString(dfile, "MIN");
  rc |= GetVector(dfile, &min);
  rc |= GetString(dfile, "MAX");
  rc |= GetVector(dfile, &max);
  rc |= GetTexture(dfile, &tex);

  rt_box(tex, min, max);

  return rc;
}

static errcode GetRing(FILE * dfile) {
  vector normal;
  vector ctr;
  void * tex;
  float a,b;
  errcode rc;

  rc = GetString(dfile, "CENTER");
  rc |= GetVector(dfile, &ctr);
  rc |= GetString(dfile, "NORMAL");
  rc |= GetVector(dfile, &normal);
  rc |= GetString(dfile, "INNER");
  fscanf(dfile, " %f ", &a);
  rc |= GetString(dfile, "OUTER");
  fscanf(dfile, " %f ", &b);
  rc |= GetTexture(dfile, &tex);

  rt_ring(tex, ctr, normal, a, b);

  return rc;
}

static errcode GetTri(FILE * dfile) {
  vector v0,v1,v2;
  void * tex;
  errcode rc;

  rc = GetString(dfile, "V0");
  rc |= GetVector(dfile, &v0);

  rc |= GetString(dfile, "V1");
  rc |= GetVector(dfile, &v1);

  rc |= GetString(dfile, "V2");
  rc |= GetVector(dfile, &v2);

  rc |= GetTexture(dfile, &tex);

  rt_tri(tex, v0, v1, v2);

  return rc;
}

static errcode GetSTri(FILE * dfile) {
  vector v0,v1,v2,n0,n1,n2;
  void * tex;
  errcode rc;

  rc = GetString(dfile, "V0");
  rc |= GetVector(dfile, &v0);

  rc |= GetString(dfile, "V1");
  rc |= GetVector(dfile, &v1);

  rc |= GetString(dfile, "V2");
  rc |= GetVector(dfile, &v2);

  rc |= GetString(dfile, "N0");
  rc |= GetVector(dfile, &n0);

  rc |= GetString(dfile, "N1");
  rc |= GetVector(dfile, &n1);

  rc |= GetString(dfile, "N2");
  rc |= GetVector(dfile, &n2);

  rc |= GetTexture(dfile, &tex);

  rt_stri(tex, v0, v1, v2, n0, n1, n2);

  return rc;
}

static errcode GetLandScape(FILE * dfile) {
  void * tex;
  vector ctr;
  apiflt wx, wy;
  int m, n;
  float a,b;
  errcode rc;

  rc = GetString(dfile, "RES");
  fscanf(dfile, "%d %d", &m, &n);

  rc |= GetString(dfile, "SCALE");
  fscanf(dfile, "%f %f", &a, &b);
  wx=a;
  wy=b;

  rc |= GetString(dfile, "CENTER");
  rc |= GetVector(dfile, &ctr);

  rc |= GetTexture(dfile, &tex);

  rt_landscape(tex, m, n, ctr, wx, wy);

  return rc;
}

static errcode GetTPolyFile(FILE * dfile) {
  void * tex;
  vector ctr, rot, scale;
  vector v1, v2, v0;
  char ifname[255];
  FILE *ifp;
  int v, totalpolys;
  RotMat RotA;
  errcode rc;

  totalpolys=0;

  rc = GetString(dfile, "SCALE");
  rc |= GetVector(dfile, &scale);

  rc |= GetString(dfile, "ROT");
  rc |= GetVector(dfile, &rot);

  degvectoradvec(&rot);
  InitRot3d(&RotA, rot.x, rot.y, rot.z);

  rc |= GetString(dfile, "CENTER");
  rc |= GetVector(dfile, &ctr);

  rc |= GetString(dfile, "FILE");
  fscanf(dfile, "%s", ifname);

  rc |= GetTexture(dfile, &tex);

  if ((ifp=fopen(ifname, "r")) == NULL) {
    fprintf(stderr, "Can't open data file %s for input!! Aborting...\n", ifname);
    return PARSEBADSUBFILE;
  }

  while (!feof(ifp)) {
    fscanf(ifp, "%d", &v);
    if (v != 3) { break; }

    totalpolys++;
    v=0;

    rc |= GetVector(ifp, &v0);
    rc |= GetVector(ifp, &v1);
    rc |= GetVector(ifp, &v2);

    Scale3d(&scale, &v0);
    Scale3d(&scale, &v1);
    Scale3d(&scale, &v2);

    Rotate3d(&RotA, &v0);
    Rotate3d(&RotA, &v1);
    Rotate3d(&RotA, &v2);

    Trans3d(&ctr, &v0);
    Trans3d(&ctr, &v1);
    Trans3d(&ctr, &v2);

    rt_tri(tex, v1, v0, v2);
  }

  fclose(ifp);

  return rc;
}
