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

#include "convgemm_blis.h"


/*
 * BLIS pack for M-->Mc
*/
void pack_RB_convgemm(char orderM, char transM, int mc, int nc, const float *restrict M, int ldM, float *restrict Mc, int RR,
             const conv_p *conv_params, int start_row, int start_col) {
    int i, j, ii, k, rr;

    if ((transM == 'N') && (orderM == 'C'))
        M = &Mcol(start_row, start_col);
    else if ((transM == 'N') && (orderM == 'R'))
        M = &Mrow(start_row, start_col);
    else if ((transM == 'T') && (orderM == 'C'))
        M = &Mcol(start_col, start_row);
    else
        M = &Mrow(start_col, start_row);

    if (((transM == 'N') && (orderM == 'C')) ||
        ((transM == 'T') && (orderM == 'R')))
        #pragma omp parallel for private(i, j, ii, rr, k)
        for (i = 0; i < mc; i += RR) {
            k = i * nc;
            rr = min(mc - i, RR);
            for (j = 0; j < nc; j++) {
                for (ii = 0; ii < rr; ii++) {
                    Mc[k] = Mcol(i + ii, j);
                    k++;
                }
                for (ii = rr; ii < RR; ii++) {
                    Mc[k] = (float) 0.0;
                    k++;
                }
                // k += (RR-rr);
            }
        }
    else
#pragma omp parallel for private(i, j, ii, rr, k)
        for (i = 0; i < mc; i += RR) {
            k = i * nc;
            rr = min(mc - i, RR);
            for (j = 0; j < nc; j++) {
                for (ii = 0; ii < rr; ii++) {
                    Mc[k] = Mcol(j, i + ii);
                    k++;
                }
                for (ii = rr; ii < RR; ii++) {
                    Mc[k] = (float) 0.0;
                    k++;
                }
                // k += (RR-rr);
            }
        }
}

/*
 * BLIS pack for M-->Mc
*/
void pack_CB_convgemm(char orderM, char transM, int mc, int nc, const float *restrict M, int ldM, float *restrict Mc, int RR,
             const conv_p *conv_params, int start_row, int start_col) {
    int i, j, jj, k, nr;

    if ((transM == 'N') && (orderM == 'C'))
        M = &Mcol(start_row, start_col);
    else if ((transM == 'N') && (orderM == 'R'))
        M = &Mrow(start_row, start_col);
    else if ((transM == 'T') && (orderM == 'C'))
        M = &Mcol(start_col, start_row);
    else
        M = &Mrow(start_col, start_row);

    k = 0;
    if (((transM == 'N') && (orderM == 'C')) ||
        ((transM == 'T') && (orderM == 'R')))
#pragma omp parallel for private(i, j, jj, nr, k)
        for (j = 0; j < nc; j += RR) {
            k = j * mc;
            nr = min(nc - j, RR);
            for (i = 0; i < mc; i++) {
                for (jj = 0; jj < nr; jj++) {
                    Mc[k] = Mcol(i, j + jj);
                    k++;
                }
                for (jj = nr; jj < RR; jj++) {
                    Mc[k] = (float) 0.0;
                    k++;
                }
                // k += (RR-nr);
            }
        }
    else
#pragma omp parallel for private(i, j, jj, nr, k)
        for (j = 0; j < nc; j += RR) {
            k = j * mc;
            nr = min(nc - j, RR);
            for (i = 0; i < mc; i++) {
                for (jj = 0; jj < nr; jj++) {
                    Mc[k] = Mcol(j + jj, i);
                    k++;
                }
                for (jj = nr; jj < RR; jj++) {
                    Mc[k] = (float) 0.0;
                    k++;
                }
                // k += (RR-nr);
            }
        }
}

