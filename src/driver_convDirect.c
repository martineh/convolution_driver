/* 
   Direct convolution 

   -----

   This program is free software: you can redistribute it and/or modify it under
   the terms of the GNU General Public License as published by the Free Software
   Foundation, either version 3 of the License, or (at your option) any later
   version.

   This program is distributed in the hope that it will be useful, but WITHOUT
   ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
   FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
   You should have received a copy of the GNU General Public License along with
   this program. If not, see <http://www.gnu.org/licenses/>.

   -----

   author    = "Enrique S. Quintana-Orti" contact   = "quintana@disca.upv.es"
   copyright = "Copyright 2021, Universitat Politecnica de Valencia"
   license   = "GPLv3"
   status    = "Production"
   version   = "1.1"
*/

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdint.h>
#include <math.h>
#include <string.h>
#include <assert.h>

#include <sys/time.h>

#include "dtypes.h"
#include "formats.h"
#include "arrays.h"
#include "sutils.h"
#include "convDirect.h"
#include "colors.h"
#include "inutils.h"

#include "im2row.h"
#include "im2col.h"

#include "modelLevel/model_level.h"
#include "gemm/gemm_blis.h"

#ifdef CONVGEMM
  #undef min
  #include "convGemm/convgemm_blis.h"
  #include "convGemm/im2row_nhwc.h"
#endif

#if defined (LOWERING) && defined (BLIS)
  #include "blis.h"
#elif defined (LOWERING) && defined (OPENBLAS)
  #include "cblas.h"
#endif

#define dabs(a)      ( (a) > 0.0 ? (a) : -(a) )

int main(int argc, char *argv[])
{
  // The definition of these matrices are not necessary as the vectorized
  // versions implicitly contain them in the corresponding codes
  // This is only necessary for the generic version based on gemm operations
  // These parameteres for the vectorized variants can be NULL  

  char* variant;
  DTYPE *D, *F, *Y, *Yg, *DT, *FT, *YT, *FB, *DEXT, *Ac, *Ctmp, *Ac_blis, *Bc_blis;
  
  size_t mc_blis, nc_blis, kc_blis;

  double t1, t2, time, tmin, error, nrm, tmp, errorthd, flops, GFLOPS;
  int    m, t,
         nmin,  nmax,  nstep,
         kmin,  kmax,  kstep,
         cmin,  cmax,  cstep,
         hmin,  hmax,  hstep,
         wmin,  wmax,  wstep,
         rmin,  rmax,  rstep,
         smin,  smax,  sstep,
         prmax, psmax, ret,
         tformat, tformatmin, tformatmax,
         n, k, c,
         h, w,
         r, s,
         pr, ps,
         in, ir, is, ic, ik, ih, iw,
         ldD1,  ldD2,  ldD3,
         ldDT1, ldDT2, ldDT3, ldDT4,
         ldF1,  ldF2,  ldF3,
         ldFB1, ldFB2, ldFB3, ldFB4,
         ldFT1, ldFT2, ldFT3, ldFT4, ldFT5,
         ldY1,  ldY2,  ldY3,
         ldYT1, ldYT2, ldYT3, ldYT4,
         visual, nreps, 
    ho, wo, homax, womax;
  
  int ib, i, i2, ii, Ci_Cib, Co_Cob, Co_Nr, Co_Mr;
  char *filename;
  FILE *fd;
  int cnn_test_num, cnn_i;
  int CIB, COB, WOB;
  size_t test_n = 0;
  //VARIABLES FOR GEMM LOWERING
  int mm, nn, kk;
  DTYPE alphap;
  DTYPE betap;
  int lda, ldb, ldc;
      
  int vpadding;
  int hpadding;
  int vstride; 
  int hstride; 
  int vdilation;
  int hdilation;
  
  #ifdef BLOCKED_TZEMENG
      WOB = MR;
      COB = NR;
      CIB = NR;
  #endif

  testConfig_t* testConf=new_CNN_Test_Config(argv);
  
  m = 2; t = 6;
  
  tformat = NHWC;
  
  #if defined(INT8)
    errorthd = 0.5;
  #elif defined(FP16)
    errorthd = 1.0e-3;
  #elif defined(FP32)
    errorthd = 1.0e-5;
  #elif defined(FP64)
    errorthd = 1.0e-14;
  #endif

    if (testConf->type == CNN_TYPE)
      fprintf(testConf->fd_csv, "l;Variant;CIB;COB;WOB;n;k;c;h;w;kh;kw;Time;GFLOPS;Error\n");    
    else
      fprintf(testConf->fd_csv, "Variant;CIB;COB;WOB;n;k;c;h;w;kh;kw;Time;GFLOPS;Error\n");    

    printf(" ==============================================================================================================================\n");
    printf(" |%s                     D R I V E R    F O R    D I R E C T    C O N V O L U T I O N    E V A L U A T I O N       %s             |\n",
	   COLOR_BOLDYELLOW, COLOR_RESET);
    printf(" ==============================================================================================================================\n");
    printf(" |  %s Variant     CIB     COB     WOB     n     k     c    h     w      kh    kw    Time     GFLOPS     Error     Evaluation%s   |\n",
	   COLOR_RESET, COLOR_RESET);
    printf(" ==============================================================================================================================\n");
    
    tmin = testConf->tmin;
        
    for (cnn_i = 0; cnn_i < testConf->cnn_num; cnn_i++) {
      
      n  = testConf->cnn[cnn_i].nmin;
      k  = testConf->cnn[cnn_i].kmin;
      c  = testConf->cnn[cnn_i].cmin;
      ho = testConf->cnn[cnn_i].homin;
      wo = testConf->cnn[cnn_i].womin;
      h  = testConf->cnn[cnn_i].hmin;
      w  = testConf->cnn[cnn_i].wmin;
      r  = testConf->cnn[cnn_i].rmin;
      s  = testConf->cnn[cnn_i].smin;

      vpadding  = 0;
      hpadding  = 0;
      vstride   = 1;
      hstride   = 1;
      vdilation = 1;
      hdilation = 1;

      //WARNING: ONLY FOR GEMM TEST; TODO: FIX THIS WITH PADDING!!
      if (r == 3) {
	h += 2;
	w += 2;
      }

      //ho = (h + 2 * vpadding - vdilation * (r - 1) - 1) / vstride + 1;
      //wo = (w + 2 * hpadding - hdilation * (s - 1) - 1) / hstride + 1;
      
      int m_gemm = n * ho * wo;
      int n_gemm = k;
      int k_gemm = c * r * s;

        #if defined(LOWERING) || defined(CONVGEMM)
	  //get_optim_mc_nc_kc(sizeof(DTYPE), m_gemm, n_gemm, k_gemm, MR, NR, &COB, &WOB, &CIB);
	  get_optim_mc_nc_kc(sizeof(DTYPE), n_gemm, m_gemm, k_gemm, NR, MR, &COB, &WOB, &CIB);
	  mc_blis = COB; nc_blis = WOB; kc_blis = CIB;
	  
          //Only BLIS Parameters
	  //mc_blis = COB = 120; 
	  //kc_blis = CIB = 640;
	  //nc_blis = WOB = 3072;

	  Ac_blis = (DTYPE *)aligned_alloc(32, TH*(MR+mc_blis)*(KR+kc_blis)*sizeof(DTYPE));
          Bc_blis = (DTYPE *)aligned_alloc(32, TH*(KR+kc_blis)*(NR+nc_blis)*sizeof(DTYPE));
        #else
	  #ifdef BLOCKED_TZEMENG
	    WOB = MR;
            COB = NR;
            CIB = NR;
          #else
	    get_optim_mc_nc_kc(sizeof(DTYPE), m_gemm, n_gemm, k_gemm, NR, MR, &COB, &WOB, &CIB);
	    //WOB=1792;
	    //COB=3072;
	    //CIB=640;
	  #endif
          if (WOB % MR != 0) {
            printf("ERROR: WOB must be multiple of MR. Now WOB=%d and MR=%d\n", WOB, MR);
            exit(-1);
          } else if (COB % NR != 0) {
            printf("ERROR: COB must be multiple of NR. Now COB=%d and NR=%d\n", COB, NR);
            exit(-1);
	  }
          //Ac = (DTYPE *) aligned_alloc( 4096, ((int) TH*ceil((WOB-1))/MR+1)*MR*CIB*sizeof(DTYPE));
          Ac = (DTYPE *) aligned_alloc( 32, ((int) TH*WOB*MR*CIB*sizeof(DTYPE)));
        #endif
      
      D = (DTYPE *) malloc( n*c*h*w*sizeof(DTYPE));
      DEXT = (DTYPE *) malloc( (h*w*n)*(r*s*c)*sizeof(DTYPE));
      
      F = (DTYPE *) malloc( k*c*r*s*sizeof(DTYPE));   
      Y = (DTYPE *) malloc( n*k*ho*wo*sizeof(DTYPE));
      
      DT = (DTYPE *) malloc( n*ceil(((float) c)/CIB)*CIB*h*w*sizeof(DTYPE));   
      FT = (DTYPE *) malloc( ceil(((float) k)/COB)*COB*ceil(((float) c)/CIB)*CIB*r*s*sizeof(DTYPE));   
      YT = (DTYPE *) malloc( n*ceil(((float) k)/COB)*COB*ho*wo*sizeof(DTYPE));   
        
      FB = (DTYPE *) malloc( ceil(((float) k)/NR)*NR*c*r*s*sizeof(DTYPE));
      Ctmp = (DTYPE *)malloc(TH * MR  * NR *sizeof(DTYPE));

      if ( testConf->test=='T' )
	Yg = (DTYPE *) malloc( n*k*ho*wo*sizeof(DTYPE) );   
      

	Ci_Cib = (int)ceil(((float) c)/CIB);
	Co_Cob = (int)ceil(((float) k)/COB);
	Co_Nr  = (int)ceil(((float) k)/NR);
	Co_Mr  = (int)ceil(((float) k)/MR);
        	

	if ( tformat == NCHW ) { // NCHW
	  ldD3 = w;
	  ldD2 = h*ldD3;
	  ldD1 = c*ldD2;
	  
	  ldF3 = s;
	  ldF2 = r*ldF3;
	  ldF1 = c*ldF2;
	  
	  ldY3 = wo;
	  ldY2 = ho*ldY3;
	  ldY1 = k*ldY2;
	  
	  //==// CONVOLUTION-FRIENDLY LAYOUT //==//
	  //NHWC MACRO DT[]
	  ldDT4 = CIB;
	  ldDT3 = w      * ldDT4;
	  ldDT2 = h      * ldDT3;      
	  ldDT1 = Ci_Cib * ldDT2;
	  //NHWC MACRO YT[]
	  ldYT4 = COB;
	  ldYT3 = wo     * ldYT4;
	  ldYT2 = ho     * ldYT3;      
	  ldYT1 = Co_Cob * ldYT2;
	  //NHWC MACRO FT[]
	  ldFT5 = COB;
	  ldFT4 = CIB    * ldFT5;
	  ldFT3 = s      * ldFT4;
	  ldFT2 = r      * ldFT3;
	  ldFT1 = Co_Cob * ldFT2;
	  //==//===========================//==//

	  ldFB4 = NR;
	  ldFB3 = c*ldFB4;
	  ldFB2 = Co_Nr*ldFB3;
	  ldFB1 = s*ldFB2;

	  generate_tensor4D( n, c, h, w, D, ldD1, ldD2, ldD3 );
	  generate_tensor4D( k, c, r, s, F, ldF1, ldF2, ldF3 );
	} 
	else { // NHWC
	  //NHWC MACRO D[] 
	  ldD3 = c;
	  ldD2 = w * ldD3;
	  ldD1 = h * ldD2;
	  //NHWC MACRO F[] 
	  ldF3 = k;
	  ldF2 = s*ldF3;
	  ldF1 = r*ldF2;
	  //NHWC MACRO Y[] 
	  ldY3 = k;
	  ldY2 = wo*ldY3;
	  ldY1 = ho*ldY2;
	  
	  //==// CONVOLUTION-FRIENDLY LAYOUT //==//
	  //NHWC MACRO DT[]
	  ldDT4 = CIB;
	  ldDT3 = w      * ldDT4;
	  ldDT2 = h      * ldDT3;      
	  ldDT1 = Ci_Cib * ldDT2;
	  //NHWC MACRO YT[]
	  ldYT4 = COB;
	  ldYT3 = wo     * ldYT4;
	  ldYT2 = ho     * ldYT3;      
	  ldYT1 = Co_Cob * ldYT2;
	  //NHWC MACRO FT[]
	  ldFT5 = COB;
	  ldFT4 = CIB    * ldFT5;
	  ldFT3 = s      * ldFT4;
	  ldFT2 = r      * ldFT3;
	  ldFT1 = Co_Cob * ldFT2;
	  //==//===========================//==//
	  
	  //NHWC MACRO FB[] 
#ifdef MK_BLIS
	  ldFB4 = MR;
	  ldFB3 = c*ldFB4;
	  ldFB2 = Co_Mr*ldFB3;
	  ldFB1 = s*ldFB2;
#else
	  ldFB4 = NR;
	  ldFB3 = c*ldFB4;
	  ldFB2 = Co_Nr*ldFB3;
	  ldFB1 = s*ldFB2;
#endif
	  
	  generate_tensor4D( n, h, w, c, D, ldD1, ldD2, ldD3 );
	  generate_tensor4D( c, r, s, k, F, ldF1, ldF2, ldF3 );
	  
	}
	
	// Set result to zeros
	for ( in=0; in<n; in++ )
	for ( ik=0; ik<k; ik++ )
	for ( ih=0; ih<ho; ih++ )
	for ( iw=0; iw<wo; iw++ ) {
	  if (tformat == NHWC)
	    Yrow_NHWC(in,ik,ih,iw) = 0.0;
	  else
	    Yrow_NCHW(in,ik,ih,iw) = 0.0;
	  
	  if ( testConf->test=='T' )
	    if (tformat == NHWC)
	      Ygrow_NHWC(in,ik,ih,iw) = 0.0;
	    else
	      Ygrow_NCHW(in,ik,ih,iw) = 0.0;
	}

    
        #if BLOCKED_TZEMENG
	for ( in=0; in<n; in++)
        for ( ih=0; ih<ho; ih++)
    	for ( iw=0; iw<wo; iw++)
    	for ( i=0,i2=0; i<k; i+=COB,i2++) {
	  ib = min(k-i, COB);
	  for ( ii=0; ii<ib; ii++)
	    YT(in, i2, ih, iw, ii) = 0.0;
	  }
        #endif

	if ( testConf->debug=='T' ) {
          if ( tformat == NCHW ) {
            print_tensor4D( "D", n, c, h, w, D, ldD1, ldD2, ldD3 );
            print_tensor4D( "F", k, c, r, s, F, ldF1, ldF2, ldF3 );
          } else {
            print_tensor4D( "D", n, h, w, c, D, ldD1, ldD2, ldD3 );
            print_tensor4D( "F", c, r, s, k, F, ldF1, ldF2, ldF3 );
          }
        }

	// Convolution
        #if BLOCKED_BLIS
	  transform_filter_block_blis(c, k,
				      r, s,
				      F,  ldF1,  ldF2,  ldF3,
				      FB, ldFB1, ldFB2, ldFB3, ldFB4, 
				      tformat);
        #elif BLOCKED_SHALOM
	  transform_filter_block_shalom(c, k,
				        r, s,
				        F,  ldF1,  ldF2,  ldF3,
				        FB, ldFB1, ldFB2, ldFB3, ldFB4, 
				        tformat);
        #elif BLOCKED_TZEMENG	
	  transform_input_tzemeng(n, c,
				  h, w, 
			          ho, wo,
				  r, s,
				  D,  ldD1,  ldD2,  ldD3,
				  DT, ldDT1, ldDT2, ldDT3, ldDT4,
				  tformat, CIB);
	
	  transform_filter_tzemeng(c, k,
				   r, s,
				   F,  ldF1,  ldF2,  ldF3,
				   FT, ldFT1, ldFT2, ldFT3, ldFT4, ldFT5,
				   tformat, CIB, COB);
        #endif      
	
	time  = 0.0; 
	t1    = dclock();
	nreps = 0;
	while ( time <= tmin ) {
	  // Convolution
          #ifdef LOWERING
	    im2row(DEXT, c * r * s, D, n, h, w, c, ho, wo, r,
	           s, 0, 0, 1, 1, 1, 1);

	    mm = k;
	    nn = ho * wo * n;
	    kk = r * s * c;
	    alphap = 1.0;
	    betap  = 0.0;
	    lda = k;
	    ldb = r * s * c;
	    ldc = k;

           #ifdef BLIS	   
	    sgemm_( "N", "N", &mm, &nn, &kk, &alphap, F, &lda,
	            DEXT, &ldb, &betap, Y, &ldc );
           #elif defined(OPENBLAS)
	     cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 
			 mm, nn, kk, alphap, F, lda, DEXT, ldb, betap, Y, ldc);
           #else
             #ifdef B3A2C0
               gemm_blis_B3A2C0( 'C', 'C', 'C', 'N', 'N', mm, nn, kk, 
               	               alphap, F, lda, DEXT, ldb, betap, Y, ldc,
                               Ac_blis, Bc_blis, mc_blis, nc_blis, kc_blis, Ctmp);
             #else
               gemm_blis_A3B2C0( 'C', 'C', 'C', 'N', 'N', mm, nn, kk, 
			       alphap, F, lda, DEXT, ldb, betap, Y, ldc,
                               Ac_blis, Bc_blis, mc_blis, nc_blis, kc_blis, Ctmp);
             #endif
           #endif

         #elif CONVGEMM
	    int vpadding  = 0;
	    int hpadding  = 0;
	    int vdilation = 1;
	    int hdilation = 1;
	    int vstride   = 1;
	    int hstride   = 1;

	    //int ho = (h + 2 * vpadding - vdilation * (r - 1) - 1) / vstride + 1;
	    //int wo = (w + 2 * hpadding - hdilation * (s - 1) - 1) / hstride + 1;

	    conv_p conv_params = { n, h, w, c, k, r, s,
	    vstride, hstride, vpadding, hpadding,
	    vdilation, hdilation, ho, wo,
	    NULL, NULL, NULL, NULL, NULL, false };

            gemm_blis_B3A2C0_orig('C', 'C', 'C',
				  'N', 'N',
				  k, ho * wo * n, r * s * c,
				  1.0, F, k,
				  D, r * s * c,
				  0.0, Y, k,
				  Ac_blis, pack_RB_convgemm,
				  Bc_blis, pack_CB_nhwc,
				  &conv_params, mc_blis, nc_blis, kc_blis, Ctmp);
         #elif RENAMED
	    convDirect_renamed(n, k, c,
			       h, w,
			       ho, wo,
			       r, s, 
			       D, ldD1, ldD2, ldD3, 
			       F, ldF1, ldF2, ldF3, 
			       Y, ldY1, ldY2, ldY3,
			       tformat);
          #elif REORDER
          convDirect_reorder(n, k, c,
			     h, w,
			     ho, wo,
			     r, s, 
			     D, ldD1, ldD2, ldD3, 
			     F, ldF1, ldF2, ldF3, 
			     Y, ldY1, ldY2, ldY3,
			     tformat);
          #elif BLOCKED
          convDirect_block(n, k, c,
			   h, w,
			   ho, wo,
			   r, s, 
			   D, ldD1, ldD2, ldD3, 
			   F, ldF1, ldF2, ldF3, 
			   Y, ldY1, ldY2, ldY3,
			   tformat, CIB, COB, WOB);
          #elif BLOCKED_BLIS
	      convDirect_block_blis(n, k, c,
	  		            h, w,
			            ho, wo,
			            r, s, 
			            D,  ldD1,  ldD2,  ldD3, 
			            FB, ldFB1, ldFB2, ldFB3, ldFB4,
			            Y,  ldY1,  ldY2,  ldY3,
                                    Ac, Ctmp,
			            tformat, CIB, COB, WOB);
          #elif BLOCKED_SHALOM
          convDirect_block_shalom(n, k, c,
			        h, w,
			        ho, wo,
			        r, s, 
			        D,  ldD1,  ldD2,  ldD3, 
			        FB, ldFB1, ldFB2, ldFB3, ldFB4,
			        Y,  ldY1,  ldY2,  ldY3,
			        tformat, CIB, COB, WOB);
          #elif BLOCKED_TZEMENG	
	      convDirect_block_tzemeng(n, k, c,
	  			       h, w,
				       ho,wo,
	  			       r, s,
				       Ctmp,
				       DT, ldDT1, ldDT2, ldDT3, ldDT4,
				       FT, ldFT1, ldFT2, ldFT3, ldFT4, ldFT5,
				       YT, ldYT1, ldYT2, ldYT3, ldYT4,
				       tformat, CIB, COB, WOB);
          #endif      
	  nreps++;
	  
	  t2   = dclock();
	  time = ( t2 > t1 ? t2 - t1 : 0.0 );
	}
	time = time/nreps;
	if ( nreps == 0 ) continue; 
	
        #if BLOCKED_TZEMENG	
	  transform_output_tzemeng(n, k,
				   h, w,
				   ho, wo,
				   r, s,
				   Y,  ldY1,  ldY2,  ldY3,
				   YT, ldYT1, ldYT2, ldYT3, ldYT4,
				   tformat, COB);
        #endif   
	  
	// Test result
	if ( testConf->test=='T' ) {
	  convDirect_original(n, k, c, h, w, ho, wo, r, s, 
			      D,  ldD1, ldD2, ldD3, 
			      F,  ldF1, ldF2, ldF3, 
			      Yg, ldY1, ldY2, ldY3,
			      tformat);
	  //print_tensor4D( "Yg", n, ho, wo, k, Yg, ldY1, ldY2, ldY3 );
	  
	  error = 0.0;
	  nrm   = 0.0;
	  for ( in=0; in<n; in++ )
	  for ( ik=0; ik<k; ik++ )
	  for ( ih=0; ih<ho; ih++ )
	  for ( iw=0; iw<wo; iw++ ) {
	    if (tformat == NHWC) {
	      tmp = (double) Ygrow_NHWC(in,ik,ih,iw);
	      nrm += tmp*tmp;
	      tmp = (double) dabs(Yrow_NHWC(in,ik,ih,iw)-Ygrow_NHWC(in,ik,ih,iw));
	      //printf("Y=%14.8e vs Yg=%14.8e\n", Yrow_NCHW(in,ik,ih,iw), Ygrow_NCHW(in,ik,ih,iw));
	      error += tmp*tmp;
	    } else {
	      tmp = (double) Ygrow_NCHW(in,ik,ih,iw);
	      nrm += tmp*tmp;
	      tmp = (double) dabs(Yrow_NCHW(in,ik,ih,iw)-Ygrow_NCHW(in,ik,ih,iw));	  
	      //printf("Y=%14.8e vs Yg=%14.8e\n", Yrow_NCHW(in,ik,ih,iw), Ygrow_NCHW(in,ik,ih,iw));
	      error += tmp*tmp;
	    }
	  }
	  if ( nrm!=0.0 )
	    error = sqrt(error) / sqrt(nrm);
	  else
	    error = sqrt(error);
	}
	else
	  error = -1.0;
        
	flops = 2.0 * n * k * c * ho * wo * r * s;
	GFLOPS  = flops / (1.0e+9 * time );
	
	if ( testConf->debug=='T' ) {
          if ( tformat == NCHW ) {
            print_tensor4D( "Ytest", n, k, h, w, Y, ldY1, ldY2, ldY3 );
            print_tensor4D( "Ycorrect", n, k, h, w, Yg, ldY1, ldY2, ldY3 );
          } else {
            print_tensor4D( "Ytest", n, h, w, k, Y, ldY1, ldY2, ldY3 );
            print_tensor4D( "Ycorrect", n, h, w, k, Yg, ldY1, ldY2, ldY3 );
          }
        }
		
	if ((test_n++ % 2) == 0)
	  printf("%s  %8s %9d %6d %7d %6d %5d %5d %5d %5d %5d %5d %10.2e %9.2e %9.2e%s",
		 COLOR_CYAN, (tformat==NCHW) ? "NCHW" : "NHWC", CIB, COB, WOB, n, k, c, h, w, r, s, time, GFLOPS, error,  COLOR_RESET);
	else
	  printf("  %8s %9d %6d %7d %6d %5d %5d %5d %5d %5d %5d %10.2e %9.2e %9.2e",
		 (tformat==NCHW) ? "NCHW" : "NHWC", CIB, COB, WOB,  n, k, c, h, w, r, s, time, GFLOPS, error);
	
	
	if ( testConf->test=='T' )
	  if ( error < errorthd )
	    printf("     %sOK%s", COLOR_GREEN, COLOR_RESET);
	  else {
	    printf("     %sERROR%s\n", COLOR_RED, COLOR_RESET);
	    exit(-1);
	  }
	else
	  printf("     %sDisabled%s", COLOR_BOLDYELLOW, COLOR_RESET);
	
    if (testConf->type == CNN_TYPE)
      fprintf(testConf->fd_csv,"%d;%s;%d;%d;%d;%d;%d;%d;%d;%d;%d;%d;%.2e;%.2f;%.2e\n",testConf->cnn[cnn_i].layer, (tformat==NCHW) ? "NCHW" : "NHWC", CIB, COB, WOB, n, k, c, h, w, r, s, time, GFLOPS, error);
    else
      fprintf(testConf->fd_csv,"%s;%d;%d;%d;%d;%d;%d;%d;%d;%d;%d;%.2e;%.2f;%.2e\n",(tformat==NCHW) ? "NCHW" : "NHWC", CIB, COB, WOB, n, k, c, h, w, r, s, time, GFLOPS, error);
		
	printf("\n");


      free(Ac); 
      #ifdef LOWERING
        free(Ac_blis); 
        free(Bc_blis);
      #endif

    /* Free data */
    free(Y);
    free(D);
    free(F);
    free(YT);
    free(FB);
    free(DEXT);
      
    if ( testConf->test=='T' ) free(Yg);

  }
    

  fclose(testConf->fd_csv);
  free_CNN_Test_Config(testConf);
    
  printf(" ==============================================================================================================================\n");

  return 0;
}
