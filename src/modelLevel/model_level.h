#ifndef MODEL_LEVEL_H
#define MODEL_LEVEL_H

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#ifndef min
  #define min(a,b)     ( (a) > (b) ? (b) : (a) )
#endif


void load_model_level_params(char *config_file, int *params);

int model_level(int isL3, int NL, int CL, int WL, int dataSize, int m, int n);

void get_optim_mc_nc_kc(int dataSize, int m, int n, int k, int mr, int nr, int *mc, int *nc, int *kc, int *params);

#endif
