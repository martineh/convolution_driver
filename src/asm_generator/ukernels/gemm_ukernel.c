
#include <stdio.h>
#include <stdlib.h>
#include <arm_neon.h>

#include "gemm_ukernels_headers.h"

void gemm_ukernel_asm(size_t mr, size_t nr, size_t _MR, size_t _NR, size_t kc, float *alpha, 
                                       float *a, float *b, float *beta, float *ctmp, 
                                       float *C, size_t ldC) {
  int i, j;
  float  beta_edge = 0.0;

  if (mr == _MR && nr == _NR) {
    gemm_ukernel_asm_4x4(kc, alpha, a, b, beta, C, ldC * sizeof(float));
  } else {
    gemm_ukernel_asm_4x4(kc, alpha, a, b, &beta_edge, ctmp, _MR * sizeof(float));
    for (j = 0; j < nr; j++)
      for (i = 0; i < mr; i++)
        C[j*ldC + i] = (*beta) * C[j*ldC + i] + ctmp[j * _MR + i];
  }
}

void pack_A( int _MR, int mc, int kc, float *A, int ldA, float *Ac) {
  int i, j, ii, k, rr;
  float32x4_t A0;
  for ( i=0; i<mc; i+=_MR ) {
    k = i * kc;
    rr = mc-i < _MR ? mc-i : _MR;
    if (rr == _MR) {
      for ( j=0; j<kc; j++ ) {
        A0 = vld1q_f32(&A[j * ldA + (i + 0)]);
        vst1q_f32(&Ac[k], A0); k += 4;
      }
    } else {
      for ( j=0; j<kc; j++ ) {
        for ( ii=0; ii < rr; ii++ ) {
          Ac[k] = A[j * ldA + (i + ii)];
          k++;
        }
        k += (_MR - rr);
      }
    }
  }
}
