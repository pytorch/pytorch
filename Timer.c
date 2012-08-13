#include "general.h"

#ifdef _MSC_VER
#include <time.h>
#else
#include <sys/time.h>
#include <sys/resource.h>
#endif

typedef struct _Timer
{
    int isRunning;

    double totalrealtime;
    double totalusertime;
    double totalsystime;

    double startrealtime;
    double startusertime;
    double startsystime;

#ifdef _MSC_VER
  time_t base_time;
#endif

} Timer;

static double torch_Timer_realtime()
{
  struct timeval current;
  gettimeofday(&current, NULL);
  return (current.tv_sec + current.tv_usec/1000000.0);
}

static double torch_Timer_usertime()
{
  struct rusage current;
  getrusage(RUSAGE_SELF, &current);
  return (current.ru_utime.tv_sec + current.ru_utime.tv_usec/1000000.0);
}

static double torch_Timer_systime()
{
  struct rusage current;
  getrusage(RUSAGE_SELF, &current);
  return (current.ru_stime.tv_sec + current.ru_stime.tv_usec/1000000.0);
}

static int torch_Timer_new(lua_State *L)
{
  Timer *timer = luaT_alloc(L, sizeof(Timer));
#ifdef _MSC_VER
  timer->base_time = 0;
  while(!timer->base_time)
    time(&timer->base_time);
#endif
  timer->isRunning = 1;
  timer->totalrealtime = 0;
  timer->totalusertime = 0;
  timer->totalsystime = 0;
  timer->startrealtime = torch_Timer_realtime();
  timer->startusertime = torch_Timer_usertime();
  timer->startsystime = torch_Timer_systime();
  luaT_pushudata(L, timer, "torch.Timer");
  return 1;
}

static int torch_Timer_reset(lua_State *L)
{
  Timer *timer = luaT_checkudata(L, 1, "torch.Timer");
  timer->totalrealtime = 0;
  timer->totalusertime = 0;
  timer->totalsystime = 0;
  timer->startrealtime = torch_Timer_realtime();
  timer->startusertime = torch_Timer_usertime();
  timer->startsystime = torch_Timer_systime();
  lua_settop(L, 1);
  return 1;
}

static int torch_Timer_free(lua_State *L)
{
  Timer *timer = luaT_checkudata(L, 1, "torch.Timer");
  luaT_free(L, timer);
  return 0;
}

static int torch_Timer_stop(lua_State *L)
{
  Timer *timer = luaT_checkudata(L, 1, "torch.Timer");
  if(timer->isRunning)  
  {
    double realtime = torch_Timer_realtime() - timer->startrealtime;
    double usertime = torch_Timer_usertime() - timer->startusertime;
    double systime = torch_Timer_systime() - timer->startsystime;
    timer->totalrealtime += realtime;
    timer->totalusertime += usertime;
    timer->totalsystime += systime;
    timer->isRunning = 0;
  }
  lua_settop(L, 1);
  return 1;  
}

static int torch_Timer_resume(lua_State *L)
{
  Timer *timer = luaT_checkudata(L, 1, "torch.Timer");
  if(!timer->isRunning)
  {
    timer->isRunning = 1;
    timer->startrealtime = torch_Timer_realtime();
    timer->startusertime = torch_Timer_usertime();
    timer->startsystime = torch_Timer_systime();
  }
  lua_settop(L, 1);
  return 1;  
}

static int torch_Timer_time(lua_State *L)
{
  Timer *timer = luaT_checkudata(L, 1, "torch.Timer");
  double realtime = (timer->isRunning ? (timer->totalrealtime + torch_Timer_realtime() - timer->startrealtime) : timer->totalrealtime);
  double usertime = (timer->isRunning ? (timer->totalusertime + torch_Timer_usertime() - timer->startusertime) : timer->totalusertime);
  double systime = (timer->isRunning ? (timer->totalsystime + torch_Timer_systime() - timer->startsystime) : timer->totalsystime);
  lua_createtable(L, 0, 3);
  lua_pushnumber(L, realtime);
  lua_setfield(L, -2, "real");
  lua_pushnumber(L, usertime);
  lua_setfield(L, -2, "user");
  lua_pushnumber(L, systime);
  lua_setfield(L, -2, "sys");
  return 1;
}

static int torch_Timer___tostring__(lua_State *L)
{
  Timer *timer = luaT_checkudata(L, 1, "torch.Timer");
  lua_pushfstring(L, "torch.Timer [status: %s]", (timer->isRunning ? "running" : "stopped"));
  return 1;
}

static const struct luaL_Reg torch_Timer__ [] = {
  {"reset", torch_Timer_reset},
  {"stop", torch_Timer_stop},
  {"resume", torch_Timer_resume},
  {"time", torch_Timer_time},
  {"__tostring__", torch_Timer___tostring__},
  {NULL, NULL}
};

void torch_Timer_init(lua_State *L)
{
  luaT_newmetatable(L, "torch.Timer", NULL, torch_Timer_new, torch_Timer_free, NULL);
  luaL_register(L, NULL, torch_Timer__);
  lua_pop(L, 1);
}
