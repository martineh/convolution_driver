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

#include <stdio.h>

#include "convgemm_blis.h"

/*
 * Computes the GEMM C := beta * C + alpha * A * B  following the BLIS approach
*/
void convgemm_blis_B3A2C0(char orderA, char orderB, char orderC,
                           char transA, char transB,
                           int m, int n, int k,
                           float alpha, const float *A, int ldA,
                           const float *B, int ldB,
                           float beta, float *C, int ldC,
                           float *Ac, pack_func pack_RB,
                           float *Bc, pack_func pack_CB,
                           const conv_p *conv_params,
			   int MC, int NC, int KC, int MR, int NR, int TH, float *Ctmp,
			   ukernel_asm ukr, ukernel_edge ukr_edge) {

    // Quick return if possible
    float zero = (float) 0.0, one = (float) 1.0;
    if ((m == 0) || (n == 0) || (((alpha == zero) || (k == 0)) && (beta == one)))
        return;

    int th_id = 0, i, j;
    float beta_edge = 0.0;
    #include "quick_gemm.h"

    for (int jc = 0; jc < n; jc += NC) {
        int nc = min(n - jc, NC);

        for (int pc = 0; pc < k; pc += KC) {
            int kc = min(k - pc, KC);
            bool last = (pc + KC) >= k;

            pack_CB(orderB, transB, kc, nc, B, ldB, Bc, NR, conv_params, pc, jc);

            float betaI = (pc == 0) ? beta : (float) 1.0;

            for (int ic = 0; ic < m; ic += MC) {
                int mc = min(m - ic, MC);

                pack_RB(orderA, transA, mc, kc, A, ldA, Ac, MR, conv_params, ic, pc);

		#ifdef OMP_ENABLE
                #pragma omp parallel for private(th_id)// collapse(2)
		#endif
                for (int jr = 0; jr < nc; jr += NR) {
		  #ifdef OMP_ENABLE
		  th_id = omp_get_thread_num();
 		  #else
		  th_id = 0;
		  #endif
		  float *Ctmp_th = &Ctmp[th_id * MR * NR];

                  for (int ir = 0; ir < mc; ir += MR) {

                    int mr = min(mc - ir, MR);
                    int nr = min(nc - jr, NR);
                    float *Cptr = (orderC == 'C') ? &Ccol(ic + ir, jc + jr) : &Crow(ic + ir, jc + jr);

		    if (mr == MR && nr == NR)
                      ukr(kc, &alpha, &Ac[ir*kc], &Bc[jr*kc], &betaI, Cptr, ldC * sizeof(float));
                    else
                      ukr_edge(mr, nr, MR, NR, kc, &alpha, &Ac[ir*kc], &Bc[jr*kc], &betaI, Ctmp_th, Cptr, ldC);

                    }
                }

            }
        }
    }
}
