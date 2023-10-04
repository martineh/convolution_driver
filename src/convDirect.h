#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "dtypes.h"
#include "formats.h"
#include "arrays.h"
#include "dtypes.h"

#include "ukernels/gemm_blis_neon_fp32.h"
#include "asm_generator/ukernels/gemm_ukernels_headers.h"

#if TH != 1
  #include <omp.h>
#endif

//#ifdef __NEON__
//#else
//  #include "gemm_blis_AVX512_fp32.h"
//#endif

#define min(a,b)     ( (a) > (b) ? (b) : (a) )

void gemm_reference( char, char, char,
                     char, char,
                     int, int, int,
	             DTYPE, DTYPE *, int,
	             DTYPE *, int,
	             DTYPE, DTYPE *, int );

/* void gemm_base( int, int, int, */
/* 		DTYPE, DTYPE *, int, */
/* 		DTYPE *, int, */
/* 		DTYPE, DTYPE *, int ); */

/* void gemm_base_col( int, int, int, */
/* 		    DTYPE, DTYPE *, int, */
/* 		    DTYPE *, int, */
/* 		    DTYPE, DTYPE *, int ); */

void im2row(float *, int, float *,
	    int, int, int, int, int, int, int, int, int,
	    int, int, int, int, int);

void convDirect_original( int, int, int,
                          int, int,
                          int, int,
                          int, int,
                          DTYPE *, int, int, int,
                          DTYPE *, int, int, int,
                          DTYPE *, int, int, int,
			  int);

void convDirect_renamed( int, int, int, 
                         int, int, 
                         int, int, 
                         int, int, 
                         DTYPE *, int, int, int, 
                         DTYPE *, int, int, int, 
                         DTYPE *, int, int, int,
			 int);

void convDirect_reorder( int, int, int, 
                         int, int, 
                         int, int, 
                         int, int, 
                         DTYPE *, int, int, int, 
                         DTYPE *, int, int, int, 
                         DTYPE *, int, int, int,
			 int);


void convDirect_block( int, int, int, 
                       int, int, 
                       int, int,
                       int, int,
                       DTYPE *, int, int, int, 
                       DTYPE *, int, int, int, 
                       DTYPE *, int, int, int,
		       int, int, int, int);


void transform_input_tzemeng( int, int, 
			      int, int, 
			      int, int, 
			      int, int,
			      DTYPE *, int, int, int,
			      DTYPE *, int, int, int, int,
			      int, int);

void transform_output_tzemeng( int, int, 
			       int, int, 
			       int, int, 
			       int, int,
			       DTYPE *, int, int, int, 
			       DTYPE *, int, int, int, int,
			       int, int);


void transform_filter_tzemeng( int, int, 
			       int, int, 
			       DTYPE *, int, int, int,
			       DTYPE *, int, int, int, int, int,
			       int, int, int);


    void convDirect_block_tzemeng( int, int, int, 
				   int, int, 
				   int, int, 
				   int, int,
			           DTYPE *,
				   DTYPE *, int, int, int, int,
				   DTYPE *, int, int, int, int, int,
				   DTYPE *, int, int, int, int,
				   int, int, int, int);

void transform_filter_block_shalom( int, int, 
				    int, int, 
				    DTYPE *, int, int, int,
				    DTYPE *, int, int, int, int,
				    int);

void convDirect_block_shalom( int, int, int, 
			      int, int, 
			      int, int, 
			      int, int,
			      DTYPE *, int, int, int, 
			      DTYPE *, int, int, int, int,
			      DTYPE *, int, int, int, 
			      int, int, int, int);


void transform_filter_block_blis( int, int, 
				  int, int, 
				  DTYPE *, int, int, int,
				  DTYPE *, int, int, int, int,
				  int);

void convDirect_block_blis( int, int, int, 
		            int, int, 
		            int, int, 
			    int, int,
			    DTYPE *, int, int, int, 
			    DTYPE *, int, int, int, int,
			    DTYPE *, int, int, int, 
			    DTYPE *, DTYPE *,
			    int, int, int, int);


void packRB( char, char, int, int, DTYPE *, int, DTYPE *, int);
