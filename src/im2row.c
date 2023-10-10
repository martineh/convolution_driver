#include "im2row.h"

void im2row(DTYPE *rows, int ld, DTYPE *in,
	    int batch, int height, int width, int channel, int oheight, int owidth,
	    int kheight, int kwidth, int vpadding, int hpadding, int vstride, int
	    hstride, int vdilation, int hdilation, int TH)
{

  int b, x, y, row, kx, ix, ky, iy, c, col;
 
  for (b = 0; b < batch; b++)
    #ifdef OMP_ENABLE
      #pragma omp parallel for private (y, row, kx, ix, ky, iy, c, col)
    #endif
    for (x = 0; x < oheight; x++)
      for (y = 0; y < owidth; y++) {
	row = b * oheight * owidth + x * owidth + y;
	for (kx = 0; kx < kheight; kx++) {
	  ix = vstride * x + vdilation * kx - vpadding;
	  if (0 <= ix && ix < height)
	    for (ky = 0; ky < kwidth; ky++) {
	      iy = hstride * y + hdilation * ky - hpadding;
	      if (0 <= iy && iy < width)
		for (c = 0; c < channel; c++) {
		  col = c * kheight * kwidth + kx * kwidth + ky;
		  rows[row * channel * kheight * kwidth + col] = in[((b * height + ix) * width + iy) * channel + c];
		}
	    }
	}
      }
  /*
  if ((kheight == 1) && (kwidth == 1)) {
    int c1, c2, c4, c5i, c6;

    c6 = channel * kheight * kwidth;
    
    for (b = 0; b < batch; b++)
      for (x = 0; x < oheight; x++) {
	ix = vstride * x - vpadding; // kx = 0;
	row = b * oheight * owidth + x * owidth;
	if (0 <= ix && ix < height) {
          for (y = 0; y < owidth; y++) {
	    iy = hstride * y - hpadding; // ky = 0
	    if (0 <= iy && iy < width) {
	      c1 = (row + y) * c6;
	      c2 = ((b * height + ix) * width + iy) * channel;
	      for (c = 0, col = 0; c < channel; c++, col++) {
	        rows[c1 + col] = in[c2 + c];
	      }
	    }
	  }
	}
      }
        
  } else if ((kheight == 3) && (kwidth == 3)) {
    int c1, c2, c3, c4, c5, c6;
    int ix0, ix1, ix2;

    c6 = channel * kheight * kwidth;

    for (b = 0; b < batch; b++) {
      for (x = 0; x < oheight; x++) {
	row = b * oheight * owidth + x * owidth;
        ix0 = vstride * x - vpadding;
	ix1 = vstride * x + vdilation - vpadding;
	ix2 = vstride * x + vdilation * 2 - vpadding;
	for (y = 0; y < owidth; y++) {
	  c1 = (row + y) * c6;
	    
	  //Iter 0 --> kx = 0 
	  if (0 <= ix0 && ix0 < height) {
	     //Iter 0: ky = 0
	     iy = hstride * y - hpadding;
	     c3 = ((b * height + ix0) * width);
	     if (0 <= iy && iy < width) { 
	       c2 = (c3 + iy) * channel;
	       for (c = 0, col = 0; c < channel; c++, col += 9)
	       	 rows[c1 + col] = in[c2 + c];
	     }
	     //Iter 1: ky = 1
	     iy = hstride * y + hdilation - hpadding;
	     if (0 <= iy && iy < width) { 
	       c2 = (c3 + iy) * channel;
	       for (c = 0, col = 1; c < channel; c++, col += 9)
	         rows[c1 + col] = in[c2 + c];
	     }
	     //Iter 2: ky = 2
	     iy = hstride * y + hdilation*2 - hpadding;
	     if (0 <= iy && iy < width) {
	       c2 = (c3 + iy) * channel;
	       for (c = 0, col = 2; c < channel; c++, col += 9)
	         rows[c1 + col] = in[c2 + c];
	     }
	   }

	   //Iter 1 --> kx = 1 
	   if (0 <= ix1 && ix1 < height) {
	     //Iter 0: ky = 0
	     iy = hstride * y - hpadding;
	     c3 = ((b * height + ix1) * width);
	     if (0 <= iy && iy < width) { 
	       c2 = (c3 + iy) * channel;
	       for (c = 0, col = kwidth; c < channel; c++, col += 9)
	       	 rows[c1 + col] = in[c2 + c];
	     }
	     //Iter 1: ky = 1
	     iy = hstride * y + hdilation - hpadding;
	     if (0 <= iy && iy < width) { 
	       c2 = (c3 + iy) * channel;
	       for (c = 0, col = kwidth + 1; c < channel; c++, col += 9)
	         rows[c1 + col] = in[c2 + c];
	     }
	     //Iter 2: ky = 2
	     iy = hstride * y + hdilation*2 - hpadding;
	     if (0 <= iy && iy < width) {
	       c2 = (c3 + iy) * channel;
	       for (c = 0, col = kwidth + 2; c < channel; c++, col += 9)
	         rows[c1 + col] = in[c2 + c];
	     }
	   }

	   //Iter 2 --> kx = 2
	   if (0 <= ix2 && ix2 < height) {
	      //Iter 0: ky = 0
	      iy = hstride * y - hpadding;
	      c3 = ((b * height + ix2) * width);
	      if (0 <= iy && iy < width) { 
	        c2 = (c3 + iy) * channel;
	        for (c = 0, col = 2 * kwidth; c < channel; c++, col += 9)
	       	  rows[c1 + col] = in[c2 + c];
	      }
	      //Iter 1: ky = 1
	      iy = hstride * y + hdilation - hpadding;
	      if (0 <= iy && iy < width) { 
	        c2 = (c3 + iy) * channel;
	        for (c = 0, col = 2 * kwidth + 1; c < channel; c++, col += 9)
	          rows[c1 + col] = in[c2 + c];
	      }
	      //Iter 2: ky = 2
	      iy = hstride * y + hdilation*2 - hpadding;
	      if (0 <= iy && iy < width) {
	        c2 = (c3 + iy) * channel;
	        for (c = 0, col = 2 * kwidth + 2; c < channel; c++, col += 9)
	          rows[c1 + col] = in[c2 + c];
	      }
	   }
         }
       }
     }
  
  } else {
    int c1, c2, c4, c5;
    c4 = kheight * kwidth;

    for (b = 0; b < batch; b++)
      for (x = 0; x < oheight; x++)
        for (y = 0; y < owidth; y++) {
	  row = b * oheight * owidth + x * owidth + y;
	  c1 = row * channel * kheight * kwidth;
	  for (kx = 0; kx < kheight; kx++) {
	    ix = vstride * x + vdilation * kx - vpadding;
	    if (0 <= ix && ix < height)
	      for (ky = 0; ky < kwidth; ky++) {
	        iy = hstride * y + hdilation * ky - hpadding;
	        c2 = ((b * height + ix) * width + iy) * channel;
	        if (0 <= iy && iy < width) {
		  col = kx * kwidth + ky;
                  //#pragma omp simd
		  for (c = 0; c < channel; c++) {
		    rows[c1 + col] = in[c2 + c];
		    col += c4;
		  }
	        }
	      }
	  }
        }
  }
*/
}

