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

#include "../dtypes.h"

void conv_winograd_workspace_alloc(int m, int r, int n, int k, int c,
              int hi, int wi, int kh, int kw,
              int vpadding, int hpadding, 
              DTYPE **U, DTYPE **V, DTYPE **M) {
    int t = m + r - 1;
    int s = m;
    int tile_h = ceil(((double) hi + 2 * vpadding - t) / s) + 1;
    int tile_w = ceil(((double) wi + 2 * hpadding - t) / s) + 1;
    *U = (DTYPE *) malloc(t * t * k * c * sizeof(DTYPE));
    *V = (DTYPE *) malloc(t * t * c * (n * tile_h * tile_w) * sizeof(DTYPE));
    *M = (DTYPE *) malloc(t * t * k * (n * tile_h * tile_w) * sizeof(DTYPE));
}

void conv_winograd_workspace_alloc_pre(int m, int r, int k, int c, DTYPE **U) {
    int t = m + r - 1;
    *U = (DTYPE *) malloc(t * t * k * c * sizeof(DTYPE));
}

void conv_winograd_workspace_alloc_kernel(int m, int r, int n, int k, int c,
              int hi, int wi, int kh, int kw,
              int vpadding, int hpadding,
              DTYPE **V, DTYPE **M) {
    int t = m + r - 1;
    int s = m;
    int tile_h = ceil(((double) hi + 2 * vpadding - t) / s) + 1;
    int tile_w = ceil(((double) wi + 2 * hpadding - t) / s) + 1;
    *V = (DTYPE *) malloc(t * t * c * (n * tile_h * tile_w) * sizeof(DTYPE));
    *M = (DTYPE *) malloc(t * t * k * (n * tile_h * tile_w) * sizeof(DTYPE));
}

void conv_winograd_workspace_dealloc(DTYPE **U, DTYPE **V, DTYPE **M) {
    free(*U); free(*V); free(*M);
}

