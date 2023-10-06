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
#include "asm_generator/ukernels/gemm_ukernels_headers.h"

#undef min
#include "convGemm/convgemm_blis.h"
#include "convGemm/im2row_nhwc.h"

#include "blis.h"
#include "cblas.h"

#define dabs(a)      ( (a) > 0.0 ? (a) : -(a) )

int main(int argc, char *argv[]) {
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
  int mm, nn, kk;
  DTYPE alphap;
  DTYPE betap;
  int lda, ldb, ldc;
  int MR, NR, TH;
  char *ALG, *GEMM;

  int vpadding;
  int hpadding;
  int vstride; 
  int hstride; 
  int vdilation;
  int hdilation;

  void (*kernel)(size_t , float *, float *, float *, float *, float *, size_t );

  testConfig_t* testConf=new_CNN_Test_Config(argv);
  m=2; t=6;

  tmin    = testConf->tmin;
  tformat = testConf->format;
  TH      = testConf->TH;
  MR = testConf->NR;
  NR = testConf->MR;
  
  ALG     = testConf->ALG; 
  GEMM    = testConf->GEMM; 
  

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

  printf(" +==================================================================================================================+\n");
  printf(" |%s                                        DRIVER FOR CONVOLUTION EVALUATION                                         %s|\n",
  COLOR_BOLDYELLOW, COLOR_RESET);
  printf(" +=========+===========================+======================================+==============================+======+\n");
  printf(" | %sMR   NR | COB(MC)  WOB(NC)  CIB(KC) |   n     k     c    h     w   (kh,kw) |  GFLOPS     Time     Error   | Test%s |\n",
  COLOR_RESET, COLOR_RESET);
  printf(" +=========+===========================+======================================+==============================+======+\n");
    
        
  for (cnn_i = 0; cnn_i < testConf->cnn_num; cnn_i++) {
    
    //-------------------------------------------------
    if (strcmp("CONVDIRECT", ALG)==0) {
      MR = testConf->NR; NR = testConf->MR;
    }

    if (MR == 4 && NR == 4)
      (kernel) = &gemm_ukernel_asm_4x4;
    else if (MR == 4 && NR == 8)
      (kernel) = &gemm_ukernel_asm_4x8;
    else if (MR == 4 && NR == 12)
      (kernel) = &gemm_ukernel_asm_4x12;
    else if (MR == 4 && NR == 16)
      (kernel) = &gemm_ukernel_asm_4x16;
    else if (MR == 4 && NR == 20)
      (kernel) = &gemm_ukernel_asm_4x20;
    else if (MR == 8 && NR == 4)
      (kernel) = &gemm_ukernel_asm_8x4;
    else if (MR == 8 && NR == 8)
      (kernel) = &gemm_ukernel_asm_8x8;
    else if (MR == 8 && NR == 12)
      (kernel) = &gemm_ukernel_asm_8x12;
    else if (MR == 12 && NR == 4)
      (kernel) = &gemm_ukernel_asm_12x4;
    else if (MR == 12 && NR == 8)
      (kernel) = &gemm_ukernel_asm_12x8;
    else if (MR == 16 && NR == 4)
      (kernel) = &gemm_ukernel_asm_16x4;
    else if (MR == 20 && NR == 4)
      (kernel) = &gemm_ukernel_asm_20x4;

    //------------------------------------------------
    
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

    if (r == 3) {h += 2; w += 2;}

    int m_gemm = n * ho * wo;
    int n_gemm = k;
    int k_gemm = c * r * s;

    if (strcmp("LOWERING", ALG)==0 || strcmp("CONVGEMM", ALG)==0) {
      get_optim_mc_nc_kc(sizeof(DTYPE), n_gemm, m_gemm, k_gemm, NR, MR, &COB, &WOB, &CIB);
      
      mc_blis = COB; nc_blis = WOB; kc_blis = CIB;

      Ac_blis = (DTYPE *)aligned_alloc(32, TH*(MR+mc_blis)*(kc_blis)*sizeof(DTYPE));
      Bc_blis = (DTYPE *)aligned_alloc(32, TH*(kc_blis)*(NR+nc_blis)*sizeof(DTYPE));
    } else {
      get_optim_mc_nc_kc(sizeof(DTYPE), m_gemm, n_gemm, k_gemm, MR, NR, &COB, &WOB, &CIB);
      MR = testConf->MR; NR = testConf->NR;
      if (WOB % MR != 0) {
        printf("ERROR: WOB must be multiple of MR. Now WOB=%d and MR=%d\n", WOB, MR);
        exit(-1);
      } else if (COB % NR != 0) {
        printf("ERROR: COB must be multiple of NR. Now COB=%d and NR=%d\n", COB, NR);
        exit(-1);
      }
      Ac = (DTYPE *) aligned_alloc( 32, ((int) TH*WOB*MR*CIB*sizeof(DTYPE)));
    }

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
      ldDT4 = CIB;
      ldDT3 = w      * ldDT4;
      ldDT2 = h      * ldDT3;      
      ldDT1 = Ci_Cib * ldDT2;
      
      ldYT4 = COB;
      ldYT3 = wo     * ldYT4;
      ldYT2 = ho     * ldYT3;      
      ldYT1 = Co_Cob * ldYT2;
      
      ldFT5 = COB;
      ldFT4 = CIB    * ldFT5;
      ldFT3 = s      * ldFT4;
      ldFT2 = r      * ldFT3;
      ldFT1 = Co_Cob * ldFT2;
      //==//===========================//==      
      ldFB4 = NR;
      ldFB3 = c*ldFB4;
      ldFB2 = Co_Nr*ldFB3;
      ldFB1 = s*ldFB2;
      generate_tensor4D( n, c, h, w, D, ldD1, ldD2, ldD3 );
      generate_tensor4D( k, c, r, s, F, ldF1, ldF2, ldF3 );

    } else { // NHWC
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

      ldFB4 = NR;
      ldFB3 = c*ldFB4;
      ldFB2 = Co_Nr*ldFB3;
      ldFB1 = s*ldFB2;
      
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
     if ( testConf->debug=='T' ) {
       if ( tformat == NCHW ) {
         print_tensor4D( "D", n, c, h, w, D, ldD1, ldD2, ldD3 );
         print_tensor4D( "F", k, c, r, s, F, ldF1, ldF2, ldF3 );
       } else {
         print_tensor4D( "D", n, h, w, c, D, ldD1, ldD2, ldD3 );
         print_tensor4D( "F", c, r, s, k, F, ldF1, ldF2, ldF3 );
       }
     }

    if (strcmp("CONVDIRECT", ALG)==0) {
      transform_filter_block_blis(c, k, r, s, F,  ldF1,  ldF2,  ldF3,
				  FB, ldFB1, ldFB2, ldFB3, ldFB4, tformat, 
				  MR, NR);
    }

    time  = 0.0; 
    t1    = dclock();
    nreps = 0;
    while ( time <= tmin ) {
      if (strcmp("LOWERING", ALG)==0) {
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

        if (strcmp("BLIS", GEMM)==0) {
	  sgemm_( "N", "N", &mm, &nn, &kk, &alphap, F, &lda, DEXT, &ldb, &betap, Y, &ldc );
	} else if (strcmp("OPENBLAS", GEMM)==0) {
	  cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 
		      mm, nn, kk, alphap, F, lda, DEXT, ldb, betap, Y, ldc);
	} else if (strcmp("B3A2C0", GEMM)==0) {
          gemm_blis_B3A2C0( 'C', 'C', 'C', 'N', 'N', mm, nn, kk, 
                            alphap, F, lda, DEXT, ldb, betap, Y, ldc,
                            Ac_blis, Bc_blis, mc_blis, nc_blis, kc_blis, MR, NR, TH, Ctmp, kernel);
	}
        
      } else if (strcmp("CONVGEMM", ALG)==0) {
        conv_p conv_params = { n, h, w, c, k, r, s,
	vstride, hstride, vpadding, hpadding,
	vdilation, hdilation, ho, wo,
	NULL, NULL, NULL, NULL, NULL, false };

        gemm_blis_B3A2C0_orig('C', 'C', 'C', 'N', 'N',
			       k, ho * wo * n, r * s * c,
			       1.0, F, k,
			       D, r * s * c,
			       0.0, Y, k,
			       Ac_blis, pack_RB_convgemm,
			       Bc_blis, pack_CB_nhwc,
			       &conv_params, mc_blis, nc_blis, kc_blis, MR, NR, TH, Ctmp);

      } else if (strcmp("CONVDIRECT", ALG)==0) {
        convDirect_block_blis(n, k, c, h, w, ho, wo, r, s, 
			      D,  ldD1, ldD2,  ldD3, 
			      FB, ldFB1, ldFB2, ldFB3, ldFB4,
			      Y,  ldY1, ldY2,  ldY3,
                              Ac, Ctmp, tformat, CIB, COB, WOB, MR, NR, TH, 
			      kernel);
      }

      nreps++;
      t2 = dclock();
      time = ( t2 > t1 ? t2 - t1 : 0.0 );

    }
    time = time/nreps;
    if ( nreps == 0 ) continue; 
	
    // Test result
    if ( testConf->test=='T' ) {
      convDirect_original(n, k, c, h, w, ho, wo, r, s, 
		          D,  ldD1, ldD2, ldD3, 
		          F,  ldF1, ldF2, ldF3, 
		          Yg, ldY1, ldY2, ldY3,
		          tformat);
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
          error += tmp*tmp;
        } else {
          tmp = (double) Ygrow_NCHW(in,ik,ih,iw);
          nrm += tmp*tmp;
          tmp = (double) dabs(Yrow_NCHW(in,ik,ih,iw)-Ygrow_NCHW(in,ik,ih,iw));	  
	  error += tmp*tmp;
	}
      }
      if ( nrm!=0.0 )
        error = sqrt(error) / sqrt(nrm);
      else
        error = sqrt(error);
    } else
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
		
    printf(" | %-3d  %-2d | %-8d %-8d %-8d| %3d %5d %5d %5d %5d   (%1d,%1d)  | %s%-10.2e%s %-8.1e %8.1e |", MR, NR, COB, WOB, CIB,  n, k, c, h, w, r, s, COLOR_BOLDWHITE, GFLOPS, COLOR_RESET, time, error);

    if ( testConf->test=='T' )
      if ( error < errorthd)
        printf("  %sOK%s  |", COLOR_GREEN, COLOR_RESET);
      else
        printf(" %sERR%s  |", COLOR_RED, COLOR_RESET);
    else
      printf("  %s-%s   |", COLOR_BOLDYELLOW, COLOR_RESET);

    printf("\n");

    fprintf(testConf->fd_csv,"%d;%d;%d;%d;%d;%d;%d;%d;%d;%d;%d;%.2e;%.2f;%.2e\n",testConf->cnn[cnn_i].layer, CIB, COB, WOB, n, k, c, h, w, r, s, time, GFLOPS, error);

    if (strcmp("LOWERING", ALG)==0 || strcmp("CONVGEMM", ALG)==0) {
      free(Ac_blis); 
      free(Bc_blis);
    } else {
      free(Ac); 
    }

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
    
  printf(" +=========+===========================+======================================+==============================+======+\n");
  printf("\n");
  return 0;
}
