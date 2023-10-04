#include "im2col.h"

void im2col(float *restrict cols, int ld, const float *restrict in, 
		int batches, int channels, int height, int width,
                 int oheight, int owidth, int kheight, int kwidth, 
		 int vpadding, int hpadding, int vstride, int hstride,
                 int vdilation, int hdilation) {

  int b, c, kx, ky, row, x, ix, y, iy, col;
  /*
  for (b = 0; b < batches; b++)
    for (c = 0; c < channels; c++)
      for (kx = 0; kx < kheight; kx++)
        for (ky = 0; ky < kwidth; ky++) {
          row = c * kheight * kwidth + kx * kwidth + ky;
            for (x = 0; x < oheight; x++) {
              ix = vstride * x + vdilation * kx - vpadding;
                if (0 <= ix && ix < height)
                  for (y = 0; y < owidth; y++) {
                    iy = hstride * y + hdilation * ky - hpadding;
                    if (0 <= iy && iy < width) {
                      col = b * oheight * owidth + x * owidth + y;
                      cols[row * batches * oheight * owidth + col] = in[((b * channels + c) * height + ix) * width + iy];
                    }
                  }
            }
        }
*/  

  int c0, c1, c2, c3, c4, c5;
  c0 = kheight * kwidth; 
  c2 = batches * oheight * owidth;

if (kheight == 1 && kwidth == 1) {
  for (b = 0; b < batches; b++) {
    c3 = b * oheight * owidth;
    for (c = 0; c < channels; c++) {
      c4 = ((b * channels + c) * height);
      row = (c * c0) * c2;
      for (x = 0, ix = 0 - vpadding; x < oheight; x++, ix += vstride) {
        if (0 <= ix && ix < height){
	  for (y = 0, iy = 0 - hpadding; y < owidth; y++, iy += hstride) {
            if (0 <= iy && iy < width) {
              col = c3 + x * owidth + y;
              cols[row + col] = in[(c4 + ix) * width + iy];
            }
          }
	}
      }
    }
  }
} else { 
  for (b = 0; b < batches; b++) {
    c3 = b * oheight * owidth;
    for (c = 0; c < channels; c++) {
      c1 = c * c0;
      c4 = ((b * channels + c) * height);
      for (kx = 0; kx < kheight; kx++)
        for (ky = 0; ky < kwidth; ky++) {
          row = (c1 + kx * kwidth + ky) * c2;
          for (x = 0, ix = vdilation * kx - vpadding; x < oheight; x++, ix += vstride) {
            if (0 <= ix && ix < height){
	      for (y = 0, iy = hdilation * ky - hpadding; y < owidth; y++, iy += hstride) {
                if (0 <= iy && iy < width) {
                  col = c3 + x * owidth + y;
                  cols[row + col] = in[(c4 + ix) * width + iy];
                }
              }
	    }
          }
        }
     }
  }
} 




}
