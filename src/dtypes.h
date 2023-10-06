#ifndef DTYPES_H
#define DTYPES_H

#ifdef FP32
#define DTYPE float
#elif FP64
#define DTYPE double
#elif INT8
#define DTYPE unsigned int
#endif


#endif
