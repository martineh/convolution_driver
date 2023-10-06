/**
 * This file is part of convGemm
 *
 * Copyright (C) 2021-22 Universitat Politècnica de València and
 *                       Universitat Jaume I
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
*/

#include <time.h>
#include <stdbool.h>
#if TH != 1
  #include <omp.h>
#endif
#include "../asm_generator/ukernels/gemm_ukernels_headers.h"

#define min(a, b) (((a)<(b))?(a):(b))
#define max(a, b) (((a)>(b))?(a):(b))

#define Acol(a1, a2)  A[ (a2)*(ldA)+(a1) ]
#define Bcol(a1, a2)  B[ (a2)*(ldB)+(a1) ]
#define Ccol(a1, a2)  C[ (a2)*(ldC)+(a1) ]
#define Mcol(a1, a2)  M[ (a2)*(ldM)+(a1) ]

#define Arow(a1, a2)  A[ (a1)*(ldA)+(a2) ]
#define Brow(a1, a2)  B[ (a1)*(ldB)+(a2) ]
#define Crow(a1, a2)  C[ (a1)*(ldC)+(a2) ]
#define Mrow(a1, a2)  M[ (a1)*(ldM)+(a2) ]

#ifdef BENCHMARK
extern double t_pack, t_kernel, t_generic;
#define BEGIN_TIMER { double t1 = get_time();
#define END_TIMER(t) double t2 = get_time(); t += t2 - t1; }
#define END_BEGIN_TIMER(t) { double t3 = get_time(); t += t3 - t1; t1 = t3; }
#else
#define BEGIN_TIMER
#define END_TIMER(t)
#define END_BEGIN_TIMER(t)
#endif

typedef struct {
    int batches, height, width, channels, kn, kheight, kwidth;
    int vstride, hstride, vpadding, hpadding, vdilation, hdilation, oheight, owidth;
    const float *bias_vector;
    const float *running_mean;
    const float *inv_std;
    const float *gamma;
    const float *beta;
    bool relu;
} conv_p;


typedef void (*pack_func)(char orderM, char transM, int mc, int nc, const float *M, int ldM, float *Mc, int RR,
                          const conv_p *conv_p, int start_row, int start_col);

typedef void (*post_func)(int mr, int nr, const float *Cc, int ldCc, float beta, float *C, int ldC,
                          const conv_p *conv_p, int start_row, int start_col, bool last);
void
gemm_blis_B3A2C0_orig(char orderA, char orderB, char orderC, char transA, char transB, int m, int n, int k, float alpha,
                      const float *A, int ldA, const float *B, int ldB,
                      float beta, float *C, int ldC, float *Ac, pack_func pack_RB, float *Bc, pack_func pack_CB,
                      const conv_p *conv_params, int MC, int NC, int KC, int MR, int NR, int TH, float *Ctmp);

void pack_RB_convgemm(char, char, int, int, const float *, int, float *, int, const conv_p *, int, int);

