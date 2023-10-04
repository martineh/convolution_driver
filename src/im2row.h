#include "dtypes.h"

#if TH != 1
  #include <omp.h>
#endif

void im2row(DTYPE *rows, int ld, DTYPE *in, int batch, 
	    int height, int width, int channel, int oheight, 
	    int owidth, int kheight, int kwidth, 
	    int vpadding, int hpadding, int vstride, int hstride, 
	    int vdilation, int hdilation);
