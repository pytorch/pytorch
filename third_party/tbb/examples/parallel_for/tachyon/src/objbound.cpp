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
 * objbound.cpp - This file contains the functions to find bounding boxes
 *              for the various primitives 
 */

#include "machine.h"
#include "types.h"
#include "macros.h"
#include "bndbox.h"

#define OBJBOUND_PRIVATE
#include "objbound.h"

static void globalbound(object ** rootlist, vector * gmin, vector * gmax) {
  vector min, max; 
  object * cur;

  if (*rootlist == NULL)  /* don't bound non-existant objects */
    return;

  gmin->x =  FHUGE;   gmin->y =  FHUGE;   gmin->z =  FHUGE;
  gmax->x = -FHUGE;   gmax->y = -FHUGE;   gmax->z = -FHUGE;

  cur=*rootlist;
  while (cur != NULL)  {  /* Go! */ 
    min.x = -FHUGE; min.y = -FHUGE; min.z = -FHUGE;
    max.x =  FHUGE; max.y =  FHUGE; max.z =  FHUGE;

    cur->methods->bbox((void *) cur, &min, &max);

    gmin->x = MYMIN( gmin->x , min.x); 
    gmin->y = MYMIN( gmin->y , min.y); 
    gmin->z = MYMIN( gmin->z , min.z); 
  
    gmax->x = MYMAX( gmax->x , max.x); 
    gmax->y = MYMAX( gmax->y , max.y); 
    gmax->z = MYMAX( gmax->z , max.z); 

    cur=(object *)cur->nextobj;
  }
}

static int objinside(object * obj, vector * min, vector * max) {
  vector omin, omax;

  if (obj == NULL)  /* non-existant object, shouldn't get here */
    return 0;

  if (obj->methods->bbox((void *) obj, &omin, &omax)) {
    if ((min->x <= omin.x) && (min->y <= omin.y) && (min->z <= omin.z) &&
        (max->x >= omax.x) && (max->y >= omax.y) && (max->z >= omax.z)) { 
      return 1;
    }
  }
  return 0;
}

static int countobj(object * root) {
  object * cur;     /* counts the number of objects on a list */
  int numobj;

  numobj=0;
  cur=root;

  while (cur != NULL) {
    cur=(object *)cur->nextobj;
    numobj++;
  } 
  return numobj;
}

static void movenextobj(object * thisobj, object ** root) {
  object * cur, * tmp;

  /* move the object after thisobj to the front of the object list  */
  /*   headed by root */
  if (thisobj != NULL) {
    if (thisobj->nextobj != NULL) {
      cur=(object *)thisobj->nextobj;            /* the object to be moved    */
      thisobj->nextobj = cur->nextobj; /* link around the moved obj */
      tmp=*root;                       /* store the root node       */
      cur->nextobj=tmp;                /* attach root to cur        */ 
      *root=cur;                       /* make cur, the new root    */
    }
  }
}

static void octreespace(object ** rootlist, int maxoctnodes) {
  object * cur;
  vector gmin, gmax, gctr;
  vector cmin1, cmin2, cmin3, cmin4, cmin5, cmin6, cmin7, cmin8;
  vector cmax1, cmax2, cmax3, cmax4, cmax5, cmax6, cmax7, cmax8;
  bndbox * box1, * box2, * box3, * box4;
  bndbox * box5, * box6, * box7, * box8;
  int skipobj;

  if (*rootlist == NULL)  /* don't subdivide non-existant data */
    return;

  skipobj=0;
  globalbound(rootlist, &gmin, &gmax);  /* find global min and max */

  gctr.x = ((gmax.x - gmin.x) / 2.0) + gmin.x;
  gctr.y = ((gmax.y - gmin.y) / 2.0) + gmin.y;
  gctr.z = ((gmax.z - gmin.z) / 2.0) + gmin.z;

  cmin1=gmin;
  cmax1=gctr;
  box1 = newbndbox(cmin1, cmax1); 

  cmin2=gmin;
  cmin2.x=gctr.x;
  cmax2=gmax;
  cmax2.y=gctr.y;
  cmax2.z=gctr.z;
  box2 = newbndbox(cmin2, cmax2); 

  cmin3=gmin;
  cmin3.y=gctr.y;
  cmax3=gmax;
  cmax3.x=gctr.x;
  cmax3.z=gctr.z;
  box3 = newbndbox(cmin3, cmax3); 

  cmin4=gmin;
  cmin4.x=gctr.x;
  cmin4.y=gctr.y;
  cmax4=gmax;
  cmax4.z=gctr.z;
  box4 = newbndbox(cmin4, cmax4); 

  cmin5=gmin;
  cmin5.z=gctr.z;
  cmax5=gctr;
  cmax5.z=gmax.z;
  box5 = newbndbox(cmin5, cmax5); 

  cmin6=gctr;
  cmin6.y=gmin.y;
  cmax6=gmax;
  cmax6.y=gctr.y;
  box6 = newbndbox(cmin6, cmax6); 

  cmin7=gctr;
  cmin7.x=gmin.x;
  cmax7=gctr;
  cmax7.y=gmax.y;
  cmax7.z=gmax.z;
  box7 = newbndbox(cmin7, cmax7); 

  cmin8=gctr;
  cmax8=gmax;
  box8 = newbndbox(cmin8, cmax8); 

  cur = *rootlist;
  while (cur != NULL)  {  
    if (objinside((object *)cur->nextobj, &cmin1, &cmax1)) {
      movenextobj(cur, &box1->objlist);  
    }  
    else if (objinside((object *)cur->nextobj, &cmin2, &cmax2)) {
      movenextobj(cur, &box2->objlist);  
    }  
    else if (objinside((object *)cur->nextobj, &cmin3, &cmax3)) {
      movenextobj(cur, &box3->objlist);  
    }  
    else if (objinside((object *)cur->nextobj, &cmin4, &cmax4)) {
      movenextobj(cur, &box4->objlist);  
    }  
    else if (objinside((object *)cur->nextobj, &cmin5, &cmax5)) {
      movenextobj(cur, &box5->objlist);  
    }  
    else if (objinside((object *)cur->nextobj, &cmin6, &cmax6)) {
      movenextobj(cur, &box6->objlist);  
    }  
    else if (objinside((object *)cur->nextobj, &cmin7, &cmax7)) {
      movenextobj(cur, &box7->objlist);  
    }  
    else if (objinside((object *)cur->nextobj, &cmin8, &cmax8)) {
      movenextobj(cur, &box8->objlist);  
    }  
    else {
      skipobj++; 
      cur=(object *)cur->nextobj;
    }
  }     

/* new scope, for redefinition of cur, and old */
  { bndbox * cur, * old;
  old=box1;
  cur=box2; 
  if (countobj(cur->objlist) > 0) {
     old->nextobj=cur;
     globalbound(&cur->objlist, &cur->min, &cur->max); 
     old=cur; 
  }      
  cur=box3;
  if (countobj(cur->objlist) > 0) {
     old->nextobj=cur;
     globalbound(&cur->objlist, &cur->min, &cur->max); 
     old=cur; 
  }      
  cur=box4;
  if (countobj(cur->objlist) > 0) {
     old->nextobj=cur;
     globalbound(&cur->objlist, &cur->min, &cur->max); 
     old=cur; 
  }      
  cur=box5;
  if (countobj(cur->objlist) > 0) {
     old->nextobj=cur;
     globalbound(&cur->objlist, &cur->min, &cur->max); 
     old=cur; 
  }      
  cur=box6;
  if (countobj(cur->objlist) > 0) {
     old->nextobj=cur;
     globalbound(&cur->objlist, &cur->min, &cur->max); 
     old=cur; 
  }      
  cur=box7;
  if (countobj(cur->objlist) > 0) {
     old->nextobj=cur;
     globalbound(&cur->objlist, &cur->min, &cur->max); 
     old=cur; 
  }      
  cur=box8;
  if (countobj(cur->objlist) > 0) {
     old->nextobj=cur;
     globalbound(&cur->objlist, &cur->min, &cur->max); 
     old=cur; 
  }      

  old->nextobj=*rootlist;

  if (countobj(box1->objlist) > 0) {
    globalbound(&box1->objlist, &box1->min, &box1->max); 
    *rootlist=(object *) box1;
  }
  else {
    *rootlist=(object *) box1->nextobj;
  }

  } /**** end of special cur and old scope */

  if (countobj(box1->objlist) > maxoctnodes) {
    octreespace(&box1->objlist, maxoctnodes);
  }
  if (countobj(box2->objlist) > maxoctnodes) {
    octreespace(&box2->objlist, maxoctnodes);
  }
  if (countobj(box3->objlist) > maxoctnodes) {
    octreespace(&box3->objlist, maxoctnodes);
  }
  if (countobj(box4->objlist) > maxoctnodes) {
    octreespace(&box4->objlist, maxoctnodes);
  }
  if (countobj(box5->objlist) > maxoctnodes) {
    octreespace(&box5->objlist, maxoctnodes);
  }
  if (countobj(box6->objlist) > maxoctnodes) {
    octreespace(&box6->objlist, maxoctnodes);
  }
  if (countobj(box7->objlist) > maxoctnodes) {
    octreespace(&box7->objlist, maxoctnodes);
  }
  if (countobj(box8->objlist) > maxoctnodes) {
    octreespace(&box8->objlist, maxoctnodes);
  }
}

void dividespace(int maxoctnodes, object **toplist) {
  bndbox * gbox;
  vector gmin, gmax;

  if (countobj(*toplist) > maxoctnodes) {
    globalbound(toplist, &gmin, &gmax);  

    octreespace(toplist, maxoctnodes); 

    gbox = newbndbox(gmin, gmax);
    gbox->objlist = NULL;
    gbox->tex = NULL; 
    gbox->nextobj=NULL;
    gbox->objlist=*toplist;
    *toplist=(object *) gbox;  
  }
}
