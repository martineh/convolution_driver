#ifndef MODEL_LEVEL_H
#define MODEL_LEVEL_H

#include <math.h>
#include <stdlib.h>
#include <stdio.h>

#if ARMV8
  //#include "archs/arm_agx_orin.h"
  #include "archs/arm_carmel.h"
#else
  #include "archs/riscv.h"
#endif

#ifndef min
  #define min(a,b)     ( (a) > (b) ? (b) : (a) )
#endif

int model_level(int isL3, int NL, int CL, int WL, int dataSize, int m, int n);
void get_optim_mc_nc_kc(int dataSize, int m, int n, int k, int mr, int nr, int *mc, int *nc, int *kc);

#endif
