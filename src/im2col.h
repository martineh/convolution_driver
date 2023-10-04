#include "dtypes.h"

void im2col(DTYPE *restrict cols, int ld, const DTYPE *restrict in, 
		int batches, int channels, int height, int width,
                 int oheight, int owidth, int kheight, int kwidth, 
		 int vpadding, int hpadding, int vstride, int hstride,
                 int vdilation, int hdilation);
