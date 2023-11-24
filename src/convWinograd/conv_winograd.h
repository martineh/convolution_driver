

extern void conv_winograd_workspace_alloc(int m, int r, int n, int k, int c,
                                          int hi, int wi, int kh, int kw,
                                          int vpadding, int hpadding,
                                          float **U, float **V, float **M);

extern void conv_winograd_workspace_dealloc(DTYPE **U, DTYPE **V, DTYPE **M);

void conv_winograd_2x2_3x3_neon_fp32_nhwc(int m, int r, int n, int k, int c,
                     int hi, int wi, int kh, int kw,
                     int vpadding, int hpadding,
                     float *D, int ldD1, int ldD2, int ldD3,
                     float *F, int ldF1, int ldF2, int ldF3,
                     float *Y, int ldY1, int ldY2, int ldY3,
                     float *biases, float *Bt, float *G, float *At,
                     float *U,  float *V, float *M,
                     const char relu, const char bn,
                     float *running_mean, float *inv_std,
                     float *gamma, float *beta);

void conv_winograd_2x2_3x3_neon_fp32_nhwc_kernel (int m, int r, int n, int k, int c,
         int hi, int wi, int kh, int kw,
         int vpadding, int hpadding,
         float *D, int ldD1, int ldD2, int ldD3,
         float *Y, int ldY1, int ldY2, int ldY3,
         float *biases, float *U, float *V, float *M,
         const char relu, const char bn,
         float *running_mean, float *inv_std,
         float *gamma, float *beta);

void conv_winograd_2x2_3x3_neon_fp32_nhwc_pre(int m, int r, int n, int k, int c, int kh, int kw,
                                              float *F, int ldF1, int ldF2, int ldF3, float *U);

