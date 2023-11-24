/**
 * This file is part of convwinograd
 *
 * An implementation of the Winograd-based convolution transform
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
#include <stdlib.h>
#include <unistd.h>
#include <stdint.h>
#include <math.h>
#include <string.h>
#include <assert.h>


#define CONV_ARGS    int m, int r, int n, int k, int c, \
                     int hi, int wi, int kh, int kw, \
                     int vpadding, int hpadding, \
                     float *D, int ldD1, int ldD2, int ldD3, \
                     float *F, int ldF1, int ldF2, int ldF3, \
                     float *Y, int ldY1, int ldY2, int ldY3, \
                     float *biases, float *Bt, float *G, float *At,\
                     float *U,  float *V, float *M, \
                     const char relu, const char bn, \
                     float *running_mean, float *inv_std, \
                     float *gamma, float *beta

#define ALLOC_ARGS   int m, int r, int n, int k, int c, \
                     int hi, int wi, int kh, int kw, \
                     int vpadding, int hpadding, \
                     float **U, float **V, float **M

#define CONV_PARAMS(v) m, r, n, k, c, \
                     h, w, r, s, \
                     vpadding, hpadding, \
                     D,  ldD1, ldD2, ldD3, \
                     F,  ldF1, ldF2, ldF3, \
                     Y, ldY1, ldY2, ldY3, \
                     NULL, Bt_ ## v , G_ ## v , At_ ## v, U,  V, M, \
                    'F', 'F', NULL, NULL, NULL, NULL


#ifdef VARIANT
#define DECL_FUNC2(v, a, f) conv_winograd_ ## v ## _ ## a ## _fp32_ ## f (CONV_ARGS)
#define DECL_FUNC(v, a, f) DECL_FUNC2(v, a, f)
#define CALL_FUNC2(v, a, f) conv_winograd_ ## v ## _ ## a ## _fp32_ ## f (CONV_PARAMS(v))
#define CALL_FUNC(v, a, f) CALL_FUNC2(v, a, f)

extern void DECL_FUNC(3x3_2x2, neon, nhwc);
extern void DECL_FUNC(2x2_3x3, neon, nhwc);
extern void DECL_FUNC(4x4_3x3, neon, nhwc);
extern void DECL_FUNC(2x2_5x5, neon, nhwc);
#endif

#else
#define CALL_FUNC2(v, a, f) conv_winograd_fp32_ ## f ## (CONV_PARAMS(v))
#define CALL_FUNC(v, a, f) CALL_FUNC2(v, a, f)
extern void conv_winograd_fp32(CONV_ARGS);
#endif

extern void conv_winograd_workspace_alloc(ALLOC_ARGS);
extern void conv_winograd_workspace_dealloc(DTYPE *U, DTYPE *V, DTYPE *M);

#define NCHW 0
#define NHWC 1

int main(int argc, char *argv[]) {
    char test;
    char *variant;
    DTYPE *D, *F, *Y, *Yg, *U, *V, *M;
    double t1, t2, time, time_alg, tmin, error, nrm, tmp, errorthd, flops, GFLOPS;
    int m, t, tmin_,
            nmin, nmax, nstep,
            kmin, kmax, kstep,
            cmin, cmax, cstep,
            hmin, hmax, hstep,
            wmin, wmax, wstep,
            rmin, rmax, rstep,
            smin, smax, sstep,
            tformat, tformatmin, tformatmax,
            n, k, c,
            h, w,
            r, s,
            in, ik, ih, iw,
            ldD1, ldD2, ldD3,
            ldF1, ldF2, ldF3,
            ldY1, ldY2, ldY3,
            visual, nreps,
            tile_H, tile_W, ho, wo, homax, womax,
            vpadding, vpaddingmin, vpaddingmax, vpaddingstep,
            hpadding, hpaddingmin, hpaddingmax, hpaddingstep;


    /*** WINOGRAD 3x3 2x2 ***/
    DTYPE Bt_3x3_2x2[16] = {1.0, 0.0, -1.0, 0.0,
                            0.0, 1.0, 1.0, 0.0,
                            0.0, -1.0, 1.0, 0.0,
                            0.0, -1.0, 0.0, 1.0};

    DTYPE G_3x3_2x2[8] = {1.0, 0.0,
                          0.5, 0.5,
                          0.5, -0.5,
                          0.0, 1.0};

    DTYPE At_3x3_2x2[12] = {1.0, 1.0, 1.0, 0.0,
                            0.0, 1.0, -1.0, 0.0,
                            0.0, 1.0, 1.0, 1.0};

    /*** WINOGRAD 2x2 3x3 ***/
    DTYPE Bt_2x2_3x3[16] = {1.0, 0.0, -1.0, 0.0,
                            0.0, 1.0, 1.0, 0.0,
                            0.0, -1.0, 1.0, 0.0,
                            0.0, 1.0, 0.0, -1.0};

    DTYPE G_2x2_3x3[12] = {1.0, 0.0, 0.0,
                           0.5, 0.5, 0.5,
                           0.5, -0.5, 0.5,
                           0.0, 0.0, 1.0};

    DTYPE At_2x2_3x3[8] = {1.0, 1.0, 1.0, 0.0,
                           0.0, 1.0, -1.0, -1.0};

    /*** WINOGRAD 4x4 3x3 ***/
    DTYPE Bt_4x4_3x3[36] = {4.0, 0.0, -5.0, 0.0, 1.0, 0.0,
                            0.0, -4.0, -4.0, 1.0, 1.0, 0.0,
                            0.0, 4.0, -4.0, -1.0, 1.0, 0.0,
                            0.0, -2.0, -1.0, 2.0, 1.0, 0.0,
                            0.0, 2.0, -1.0, -2.0, 1.0, 0.0,
                            0.0, 4.0, 0.0, -5.0, 0.0, 1.0};

    DTYPE G_4x4_3x3[18] = {1.0 / 4.0, 0.0, 0.0,
                           -1.0 / 6.0, -1.0 / 6.0, -1.0 / 6.0,
                           -1.0 / 6.0, 1.0 / 6.0, -1.0 / 6.0,
                           1.0 / 24.0, 1.0 / 12.0, 1.0 / 6.0,
                           1.0 / 24.0, -1.0 / 12.0, 1.0 / 6.0,
                           0.0, 0.0, 1.0};

    DTYPE At_4x4_3x3[24] = {1.0, 1.0, 1.0, 1.0, 1.0, 0.0,
                            0.0, 1.0, -1.0, 2.0, -2.0, 0.0,
                            0.0, 1.0, 1.0, 4.0, 4.0, 0.0,
                            0.0, 1.0, -1.0, 8.0, -8.0, 1.0};

    /*** WINOGRAD 2x2 5x5 ***/
    DTYPE Bt_2x2_5x5[36] = {4.0, 0.0, -5.0, 0.0, 1.0, 0.0,
                            0.0, -4.0, -4.0, 1.0, 1.0, 0.0,
                            0.0, 4.0, -4.0, -1.0, 1.0, 0.0,
                            0.0, -2.0, -1.0, 2.0, 1.0, 0.0,
                            0.0, 2.0, -1.0, -2.0, 1.0, 0.0,
                            0.0, 4.0, 0.0, -5.0, 0.0, 1.0};

    DTYPE G_2x2_5x5[30] = {1.0 / 4.0, 0.0, 0.0, 0.0, 0.0,
                           -1.0 / 6.0, -1.0 / 6.0, -1.0 / 6.0, -1.0 / 6.0, -1.0 / 6.0,
                           -1.0 / 6.0, 1.0 / 6.0, -1.0 / 6.0, 1.0 / 6.0, -1.0 / 6.0,
                           1.0 / 24.0, 1.0 / 12.0, 1.0 / 6.0, 1.0 / 3.0, 2.0 / 3.0,
                           1.0 / 24.0, -1.0 / 12.0, 1.0 / 6.0, -1.0 / 3.0, 2.0 / 3.0,
                           0.0, 0.0, 0.0, 0.0, 1.0};

    DTYPE At_2x2_5x5[12] = {1.0, 1.0, 1.0, 1.0, 1.0, 0.0,
                            0.0, 1.0, -1.0, 2.0, -2.0, 1.0};


    m = 2;
    t = 6;


#if defined(FP16)
    errorthd = 1.0e-3;
#elif defined(FP32)
    errorthd = 1.0e-5;
#elif defined(FP64)
    errorthd = 1.0e-14;
#endif

//#define m2r3

                                     s = r;
                                        //hpadding = vpadding;
                                        // Generate random data
                                        // printf("# -->Generate data\n"); fflush(stdout);
                                        ho = (h + 2 * vpadding - r) / 1 + 1;
                                        wo = (w + 2 * hpadding - s) / 1 + 1;
#if defined(m2r3)
                                        m = 2;
#else
                                        m = 4;
#endif
                                        t = m + r - 1;
                                        tile_H = ceil(((double) h + 2 * vpadding - t) / m) + 1;
                                        tile_W = ceil(((double) w + 2 * hpadding - t) / m) + 1;
                                        conv_winograd_workspace_alloc(m, r, n, k, c, h, w, r, s, vpadding, hpadding, &U, &V, &M);

                                            // NHWC
                                            ldD3 = c;
                                            ldD2 = w * ldD3;
                                            ldD1 = h * ldD2;

                                            ldF3 = k;
                                            ldF2 = s * ldF3;
                                            ldF1 = r * ldF2;

                                            ldY3 = k;
                                            ldY2 = wo * ldY3;
                                            ldY1 = ho * ldY2;

                                            generate_tensor4D(n, h, w, c, D, ldD1, ldD2, ldD3);
                                            generate_tensor4D(c, r, s, k, F, ldF1, ldF2, ldF3);

                                        memset(Y, 0, nmax * kmax * homax * womax * sizeof(DTYPE));
                                        memset(Yg, 0, nmax * kmax * homax * womax * sizeof(DTYPE));
                                        // printf("# -->Solve problem\n"); fflush(stdout);

                                        time = 0.0;
                                        t1 = dclock();
                                        nreps = 0;
                                        while (time <= tmin) {
                                            // Winograd
                                            if (strcmp(variant, "WINGRD\0") == 0) {
                                                if (r == 2 && s == 2) {
                                                    m = 3;
                                                    tformat == NCHW ?
                                                    CALL_FUNC(3x3_2x2, VARIANT, nchw) :
                                                    CALL_FUNC(3x3_2x2, VARIANT, nhwc);
                                                } else if (r == 3 && s == 3) {
#if defined(m2r3)
                                                    m = 2;
                                                    tformat == NCHW ?
                                                    CALL_FUNC(2x2_3x3, VARIANT, nchw) :
                                                    CALL_FUNC(2x2_3x3, VARIANT, nhwc);
#else
                                                    m = 4;
                                                    tformat == NCHW ?
                                                    CALL_FUNC(4x4_3x3, VARIANT, nchw) :
                                                    CALL_FUNC(4x4_3x3, VARIANT, nhwc);
#endif
                                                } else if (r == 5 && s == 5) {
                                                    m = 2;
                                                    tformat == NCHW ?
                                                    CALL_FUNC(2x2_5x5, VARIANT, nchw) :
                                                    CALL_FUNC(2x2_5x5, VARIANT, nhwc);
                                                } else break;
                                            } else {
                                                printf("Error: Unknown variant %s\n", variant);
                                                exit(-1);
                                            }
                                            nreps++;

                                            t2 = dclock();
                                            time = (t2 > t1 ? t2 - t1 : 0.0);
                                        }
                                        time_alg = time / nreps;
                                        if (nreps == 0) continue;

#if __x86_64__ && __LP64__
                                        memset(U, 0, t * t * k * c * sizeof(DTYPE));
                                        memset(V, 0, t * t * c * (n * tile_H * tile_W) * sizeof(DTYPE));
                                        memset(M, 0, t * t * k * (n * tile_H * tile_W) * sizeof(DTYPE));
                                        memset(Y, 0, nmax * kmax * homax * womax * sizeof(DTYPE));
                                        if (test == 'T')
                                            memset(Yg, 0, nmax * kmax * homax * womax * sizeof(DTYPE));

                                        time = 0.0;
                                        t1 = dclock();
                                        nreps = 0;
                                        while (time <= tmin) {
                                            // Winograd
                                            if (strcmp(variant, "WINGRD\0") == 0) {
                                                if (r == 3 && s == 3) {
                                                    m = 4;
                                                    tformat == NCHW ?
                                                    CALL_FUNC(4x4_3x3, VARIANT, nchw) :
                                                    CALL_FUNC(4x4_3x3, VARIANT, nhwc);
#endif
                                                } else if (r == 5 && s == 5) {
                                                    m = 2;
                                                    tformat == NCHW ?
                                                    CALL_FUNC(2x2_5x5, VARIANTAVX, nchw) :
                                                    CALL_FUNC(2x2_5x5, VARIANTAVX, nhwc);
                                                } else break;
                                            } else {
                                                printf("Error: Unknown variant %s\n", variant);
                                                exit(-1);
                                            }
                                            nreps++;

                                            t2 = dclock();
                                            time = (t2 > t1 ? t2 - t1 : 0.0);
                                        }
                                        time_avx = time / nreps;
                                        if (nreps == 0) continue;

                                        memset(U, 0, t * t * k * c * sizeof(DTYPE));
                                        memset(V, 0, t * t * c * (n * tile_H * tile_W) * sizeof(DTYPE));
                                        memset(M, 0, t * t * k * (n * tile_H * tile_W) * sizeof(DTYPE));
                                        memset(Y, 0, nmax * kmax * homax * womax * sizeof(DTYPE));
                                        if (test == 'T')
                                            memset(Yg, 0, nmax * kmax * homax * womax * sizeof(DTYPE));

                                        time = 0.0;
                                        t1 = dclock();
                                        nreps = 0;
                                        while (time <= tmin) {
                                            // Winograd
                                            if (strcmp(variant, "WINGRD\0") == 0) {
                                                if (r == 3 && s == 3) {
#if defined(m2r3)
                                                    m = 2;
                                                    tformat == NCHW ?
                                                    CALL_FUNC(2x2_3x3, VARIANT, nchw) :
                                                    CALL_FUNC(2x2_3x3, VARIANT, nhwc);
#else
                                                    m = 4;
                                                    tformat == NCHW ?
                                                    CALL_FUNC(4x4_3x3, VARIANT, nchw) :
                                                    CALL_FUNC(4x4_3x3, VARIANT, nhwc);
#endif
                                                } else if (r == 5 && s == 5) {
                                                    m = 2;
                                                    tformat == NCHW ?
                                                    CALL_FUNC(2x2_5x5, VARIANTAVX512, nchw) :
                                                    CALL_FUNC(2x2_5x5, VARIANTAVX512, nhwc);
                                                } else break;
                                            } else {
                                                printf("Error: Unknown variant %s\n", variant);
                                                exit(-1);
                                            }
                                            nreps++;

                                            t2 = dclock();
                                            time = (t2 > t1 ? t2 - t1 : 0.0);
                                        }
                                        time_avx512 = time / nreps;
                                        if (nreps == 0) continue;
#endif
                                        // Test result
                                        if (test == 'T') {
                                            convDirect(n, k, c,
                                                       h, w,
                                                       r, s,
                                                       vpadding, hpadding,
                                                       D, ldD1, ldD2, ldD3,
                                                       F, ldF1, ldF2, ldF3,
                                                       Yg, ldY1, ldY2, ldY3,
                                                       tformat);

                                            error = 0.0;
                                            nrm = 0.0;
                                            for (in = 0; in < n; in++)
                                                for (ik = 0; ik < k; ik++)
                                                    for (ih = 0; ih < ho; ih++)
                                                        for (iw = 0; iw < wo; iw++) {
                                                            if (tformat == NCHW) {
                                                                tmp = (double) Ygrow_nchw(in, ik, ih, iw);
                                                                nrm += tmp * tmp;
                                                                tmp = (double) dabs(Yrow_nchw(in, ik, ih, iw) -
                                                                                    Ygrow_nchw(in, ik, ih, iw));
                                                                error += tmp * tmp;
                                                            } else {
                                                                tmp = (double) Ygrow_nhwc(in, ik, ih, iw);
                                                                nrm += tmp * tmp;
                                                                tmp = (double) dabs(Yrow_nhwc(in, ik, ih, iw) -
                                                                                    Ygrow_nhwc(in, ik, ih, iw));
                                                                error += tmp * tmp;
                                                            }
                                                        }
                                            if (nrm != 0.0)
                                                error = sqrt(error) / sqrt(nrm);
                                            else
                                                error = sqrt(error);
                                        } else
                                            error = -1.0;

                                        // Print results
                                        if (visual == 1) {
                                            if (tformat == NCHW) {
                                                print_tensor4D("Yc", n, k, h, w, Y, ldY1, ldY2, ldY3);
                                                print_tensor4D("Ycd", n, k, h, w, Yg, ldY1, ldY2, ldY3);
                                            } else {
                                                print_tensor4D("Yc", n, h, w, k, Y, ldY1, ldY2, ldY3);
                                                print_tensor4D("Ycd", n, h, w, k, Yg, ldY1, ldY2, ldY3);
                                            }
                                        }

                                        //printf("-->Results\n");
                                        //printf("   Time         = %12.6e seg.\n", time  );
                                        flops = 2.0 * n * k * c * h * w * r * s;
                                        GFLOPS = flops / (1.0e+9 * time_alg);
                                        //printf("   GFLOPs       = %12.6e     \n", GFLOPS  );

#if __x86_64__ && __LP64__
                                        printf("      %6s %5d %5d %5d %5d %5d %5d %5d %5d %5d %7s %10.2e %10.2e %12.2e %8.2f %11.2f %9.2e %9.2e",
                                               variant, n, k, c, h, w, r, s, vpadding, hpadding,
                                               (tformat == NCHW) ? "NCHW" : "NHWC", time_alg, time_avx, time_avx512, time_alg/time_avx, time_alg/time_avx512, GFLOPS, error);
                                        speedupavx_avg = (speedupavx_avg * cnt + time_alg/time_avx) / (cnt + 1);
                                        speedupavx512_avg = (speedupavx512_avg * cnt + time_alg/time_avx512) / (cnt + 1);
                                        cnt++;
#else
                                        printf("      %6s %5d %5d %5d %5d %5d %5d %5d %5d %5d %7s %10.2e %9.2f %9.2e",
                                               variant, n, k, c, h, w, r, s, vpadding, hpadding,
                                               (tformat == NCHW) ? "NCHW" : "NHWC", time_alg, GFLOPS, error);
#endif
                                        if (error < errorthd)
                                            printf("    [OK]");
                                        else
                                            printf("  ******");
                                        printf("\n");

                                        conv_winograd_workspace_dealloc(&U, &V, &M);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    /* Free data */
    free(Y);
    free(D);
    free(F);

    if (test == 'T')
        free(Yg);
    printf("# End of program...\n");
    printf("# ========================================================================================================");
    if (test == 'T') printf("=======");
    printf("\n");
#if __x86_64__ && __LP64__
    printf("Speedup AVX256: %10.2f\n", speedupavx_avg);
    printf("Speedup AVX512: %10.2f\n", speedupavx512_avg);
#endif

    return 0;
}
