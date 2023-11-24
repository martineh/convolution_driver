#include "convDirect.h"

void gemm_base( int m, int n, int k, 
                DTYPE alpha, DTYPE *A, int ldA, 
                             DTYPE *B, int ldB, 
                DTYPE beta,  DTYPE *C, int ldC ){

  //Baseline micro-kernel 
  //Replace with specialized micro-kernel where C-->m x n is resident in registers

  int    i, j, p;
  DTYPE  tmp;

  for ( j=0; j<n; j++ )
    for ( i=0; i<m; i++ ) {
      tmp = 0.0; 
      for ( p=0; p<k; p++ ) 
        tmp += Arow(i,p) * Brow(p,j);
      Crow(i,j) = alpha*tmp + beta*Crow(i,j);
    }
}

void gemm_base_col( int m, int n, int k, 
		    DTYPE alpha, DTYPE *A, int ldA, 
		    DTYPE *B, int ldB, 
		    DTYPE beta,  DTYPE *C, int ldC ) {

  //Baseline micro-kernel 
  //Replace with specialized micro-kernel where C-->m x n is resident in registers

  int    i, j, p;
  DTYPE  tmp;

  for ( j=0; j<n; j++ )
    for ( i=0; i<m; i++ ) {
      tmp = 0.0; 
      for ( p=0; p<k; p++ ) 
        tmp += Acol(i,p) * Bcol(p,j);
      Ccol(i,j) = alpha*tmp + beta*Ccol(i,j);
    }
}


void packRB( char orderA, char transA, int mc, int nc, DTYPE *A, int ldA, DTYPE *Ac, int RR ){
  int    i, j, ii, k, rr;

  if ( ((transA=='N')&&( orderA=='C'))||
       ((transA=='T')&&( orderA=='R')) )
    for ( i=0; i<mc; i+=RR ) { 
      k = i*nc;
      rr = min( mc-i, RR );
      for ( j=0; j<nc; j++ ) {
        for ( ii=0; ii<rr; ii++ ) {
          Ac[k] = Acol(i+ii,j);
          k++;
        }
        k += (RR-rr);
      }
    }
  else
    for ( i=0; i<mc; i+=RR ) { 
      k = i*nc;
      rr = min( mc-i, RR );
      for ( j=0; j<nc; j++ ) {
        for ( ii=0; ii<rr; ii++ ) {
           Ac[k] = Acol(j,i+ii);
          k++;
        }
        k += (RR-rr);
      }
    }
}


void convDirect_original( int n, int k, int c, 
                          int h, int w, 
                          int ho, int wo, 
                          int r, int s, 
			  int vpadding, int hpadding,
                          DTYPE *D, int ldD1, int ldD2, int ldD3,
	                  DTYPE *F, int ldF1, int ldF2, int ldF3,
                          DTYPE *Yg, int ldY1, int ldY2, int ldY3,
			  int tformat)
{ 
  int in, ik, ic, ih, iw, ir, is, x_x, x_y;

  // Quick return if possible
  if ( (n==0)||(k==0)||(c==0)||
       (h==0)||(w==0)||
       (r==0)||(s==0))
    return;


  //ho = floor(((double) h - r) / 1) + 1;
  //wo = floor(((double) w - s) / 1) + 1;
  if (tformat == NHWC) {
    for ( in=0;  in<n;   in++ ) 
    for ( ik=0;  ik<k;   ik++ ) 
    for ( ic=0;  ic<c;   ic++ ) 
    for ( ih=0;  ih<ho;  ih++ ) 
    for ( iw=0;  iw<wo;  iw++ ) 
    for ( ir=0;  ir<r;   ir++ ) {
      x_x = ih + ir  - vpadding;
      if (0 <= x_x && x_x < h) 
	for ( is=0; is<s; is++ ) {
	  x_y = iw + is - hpadding;
	  if (0 <= x_y && x_y < w) {
	    //printf("D[%d]=%.4f\n", in+ic+x_x+x_y, Drow_NHWC(in,ic,x_x,x_y));
	    Ygrow_NHWC(in,ik,ih,iw) += Drow_NHWC(in,ic,x_x,x_y) * Frow_NHWC(ik,ic,ir,is);        
          }
	}
    }
  } else {
    for ( in=0;  in<n;   in++ ) 
    for ( ik=0;  ik<k;   ik++ ) 
    for ( ic=0;  ic<c;   ic++ ) 
    for ( ih=0;  ih<ho;  ih++ ) 
    for ( iw=0;  iw<wo;  iw++ ) 
    for ( ir=0;  ir<r;   ir++ ) {
      x_x = ih + ir;
      if (0 <= x_x && x_x < h) 
	for ( is=0; is<s; is++ ) {
	  x_y = iw + is;
	  if (0 <= x_y && x_y < w)
	    Ygrow_NCHW(in,ik,ih,iw) += Drow_NCHW(in,ic,x_x,x_y) * Frow_NCHW(ik,ic,ir,is);        
	}
    }
  }

}  

void transform_filter_block_blis( int Ci, int Co,
				  int Hf, int Wf,
				  DTYPE *F,  int ldF1,  int ldF2,  int ldF3,
				  DTYPE *FB, int ldFB1, int ldFB2, int ldFB3, int ldFB4,
				  int tformat, int MR, int NR) {
  int  i, j, jj, jb, j2, m, n;  
  // Quick return if possible
  if ( (Ci==0)||(Co==0)||
       (Hf==0)||(Wf==0) )
    return;

  if (tformat == NHWC) {
    for ( j=0,j2=0; j<Co; j+=NR,j2++ ) {
      jb = min(Co-j, NR);
      for ( i=0; i<Ci; i++ )
	for ( n=0; n<Hf; n++ )
	  for ( m=0; m<Wf; m++ )
	    for ( jj=0; jj<jb; jj++ ) {
              FBrow_NHWC(j2, i, n, m, jj) = Frow_NHWC(j+jj, i, n, m);
	    }
    }
  } else {
    printf("Case not yet implemented!\n");
    exit(-1);
  }

}

void convDirect_block_blis( int t,     int Co,   int Ci, 
                            int Ho,    int Wo, 
                            int ho,    int wo, 
                            int Hf,    int Wf,
		            DTYPE *D,  int ldD1,  int ldD2,  int ldD3,
	                    DTYPE *FB, int ldFB1, int ldFB2, int ldFB3, int ldFB4,
                            DTYPE *Y,  int ldY1,  int ldY2,  int ldY3,
                            DTYPE *Ac, DTYPE *Ctmp,
		            int tformat, int CIB, int COB, int WOB, 
			    int MR, int NR, int TH, ukernel_asm ukr, ukernel_edge ukr_edge) {

  if (tformat == NCHW) {
    printf("1. Case not yet implemented %d\n", tformat); 
    exit(-1); 
  }
  
  // Quick return if possible
  if ( (t==0) ||(Co==0)||(Ci==0)||
       (Ho==0)||(Wo==0)||
       (Hf==0)||(Wf==0) )
    return;

  int h, i, j, k, l, m, n, i2, j2,
      ii, jj, kk, ib, jb, kb, Cob_Nr = COB/NR, Cob_Mr = COB/MR;

  int ju, iu;

  int jr, nr, jr2, ir, mr, in = 0;
  int kb_limit;
  float alpha = 1.0;
  float beta  = 1.0;
  float beta_edge = 0.0;

  DTYPE *Y_ptr;

  if (TH == 1) {
    for ( h=0; h<t; h++ ) 
       for ( i=0,i2=0; i<Ci; i+=CIB,i2++ ) { 
         ib = min(Ci-i, CIB); 
         for ( l=0; l<ho; l++ ) {
           for ( k=0; k<wo; k+=WOB ) { 
             kb = min(wo-k, WOB); 
             for ( n=0; n<min(Hf,Ho-l); n++ ) {
               for ( m=0; m<Wf; m++ ) {
                 packRB( 'R', 'N', kb, ib, &Drow_NHWC(h, i, l+n, k+m), ldD3, Ac, MR);
		 //packRB_neon('R', 'N', kb, ib, &Drow_NHWC(h, i, l + n, k + m), ldD3, Ac, MR);
		 for ( j=0,j2=0; j<Co; j+=COB,j2++ ) { 
		   jb = min(Co-j, COB); 
		   for ( jr=0, jr2=0; jr < jb; jr += NR, jr2++) {
		     nr = min(jb-jr, NR);
		     for ( ir=0; ir < min(kb, Wo-k-m+1); ir += MR) {
		       mr = min(min(kb, Wo-k-m+1)-ir, MR);
		       Y_ptr=&Yrow_NHWC(h, j + jr, l, k + ir);

		       if (mr == MR && nr == NR)
                         ukr(ib, &alpha, &FBrow_NHWC(j2 * Cob_Nr + jr2, i, n, m, 0), &Ac[ir*ib], &beta, Y_ptr, ldY3 * sizeof(float));
                       else
                         ukr_edge(nr, mr, NR, MR, ib, &alpha, &FBrow_NHWC(j2 * Cob_Nr + jr2, i, n, m, 0), &Ac[ir*ib], &beta, Ctmp, Y_ptr, ldY3);

                     }
                   }
                 }
	       }
	     } 
           } 
         }
       }
  } else { 
    int ho_chunk = (int) ceil((double)ho/TH); 
    int th_id = 0;
    DTYPE *Ctmp_ptr = Ctmp;
    #ifdef OMP_ENABLE
    #pragma omp parallel num_threads(TH) private(th_id, h, i, i2, ib, l, k, kb, n, m, j, j2, jb, jr, nr, jr2, ir, mr, kb_limit, ju, iu, Y_ptr, Ctmp_ptr) firstprivate(ho_chunk, beta_edge)
    {
      th_id = omp_get_thread_num();
      DTYPE *Ctmp_ptr = &Ctmp[th_id * (MR * NR)];
    #else
      printf("ERROR: Parallel option configured but not compiled\n"); 
      exit(-1);
    #endif

      for ( h=0; h<t; h++ ) {
        for ( i=0,i2=0; i<Ci; i+=CIB,i2++ ) { 
          ib = min(Ci-i, CIB); 
          for ( l=ho_chunk*th_id; l< min(ho_chunk*th_id + ho_chunk, ho); l++ ) 
            for ( k=0; k<wo; k+=WOB ) { 
              kb = min(wo-k, WOB); 
              for ( n=0; n<min(Hf,Ho-l); n++ ) {
                for ( m=0; m < Wf; m++ ) {
                  packRB( 'R', 'N', kb, ib, &Drow_NHWC(h, i, l+n, k+m), ldD3, &Ac[th_id * CIB * WOB], MR);
		  kb_limit = min(kb, Wo-k-m+1);
	          for ( j=0,j2=0; j<Co; j+=COB,j2++ ) { 
	            jb = min(Co-j, COB);
	            for (jr = 0; jr < jb; jr += NR) {
	              nr = min(jb-jr, NR);
	              jr2 = jr/NR;
	              for ( ir=0; ir < kb_limit; ir += MR) {
	  	        mr = min(kb_limit-ir, MR);
		        Y_ptr=&Yrow_NHWC(h, j + jr, l, k + ir);
                        
		       if (mr == MR && nr == NR)
                         ukr(ib, &alpha, &FBrow_NHWC(j2 * Cob_Nr + jr2, i, n, m, 0), 
			    &Ac[(th_id * CIB * WOB) + (ir*ib)], &beta, Y_ptr, ldY3 * sizeof(float));
                       else
                         ukr_edge(nr, mr, NR, MR, ib, &alpha, &FBrow_NHWC(j2 * Cob_Nr + jr2, i, n, m, 0), 
				&Ac[(th_id * CIB * WOB) + (ir*ib)], &beta, Ctmp_ptr, Y_ptr, ldY3);

	              }
                    }
                  }
                }
              } 
            } 
          } 
      }
    }
  #ifdef OMP_ENABLE
  }  
  #endif
}


/*
//NEW PARALLELIZATION OVER Co-Wo loop
void convDirect_block_blis( int t,     int Co,   int Ci, 
                            int Ho,    int Wo, 
                            int ho,    int wo, 
                            int Hf,    int Wf,
		            DTYPE *D,  int ldD1,  int ldD2,  int ldD3,
	                    DTYPE *FB, int ldFB1, int ldFB2, int ldFB3, int ldFB4,
                            DTYPE *Y,  int ldY1,  int ldY2,  int ldY3,
                            DTYPE *Ac, DTYPE *Ctmp,
		            int tformat, int CIB, int COB, int WOB, int MR, int NR) { 

  if (tformat == NCHW) {
    printf("1. Case not yet implemented %d\n", tformat); 
    exit(-1); 
  }
  
  // Quick return if possible
  if ( (t==0) ||(Co==0)||(Ci==0)||
       (Ho==0)||(Wo==0)||
       (Hf==0)||(Wf==0) )
    return;

  int h, i, j, k, l, m, n, i2, j2,
      ii, jj, kk, ib, jb, kb, Cob_Nr = COB/NR, Cob_Mr = COB/MR;
      //DTYPE Cc[MR*NR], blis_beta = 0.0;

  int jr, nr, jr2, ir, mr, in = 0;
  float alpha = 1.0;
  float beta  = 1.0;


  int jb_limit, kb_limit, kb2, ir_base; 

  #pragma omp parallel num_threads(TH) private(h, i, i2, ib, l, k, kb, n, m, j, j2, jb, jr, nr, jr2, ir, mr, kb2, kb_limit, jb_limit, ir_base)
  {
    int th_id = omp_get_thread_num();
    for ( h=0; h<t; h++ ) 
      for ( i=0,i2=0; i<Ci; i+=CIB,i2++ ) { 
        ib = min(Ci-i, CIB); 
        for ( l=0; l<ho; l++ ) 
          for ( k=0; k<wo; k+=WOB ) { 
            kb = min(wo-k, WOB); 
	    int its_kb = (int) ceil((double)kb/MR/TH);
            for ( n=0; n<min(Hf,Ho-l); n++ ) {
              for ( m=0; m < Wf; m++ ) {
		DTYPE *D_tmp = &Drow_NHWC(h, i, l+n, k+m);
                packRB( 'R', 'N', min(its_kb * MR, kb - its_kb * MR * th_id), ib, D_tmp + its_kb * MR * th_id * ldD3, ldD3, Ac + ib * its_kb * MR * th_id, MR);
                #pragma omp barrier
		kb2 = min(kb, Wo-k-m+1);
	        for ( j=0,j2=0; j<Co; j+=COB,j2++ ) { 
	          jb = min(Co-j, COB);
  		  int jb_chunk = (int) ceil((double)jb/NR/TH);
  		  int jr_chunk = (int) ceil((double)kb2/MR/TH);
		  if (jb_chunk > jr_chunk) {
		    jr = th_id * jb_chunk * NR;
		    jb_limit = min(th_id * jb_chunk * NR + jb_chunk * NR, jb);

		    ir_base = 0;
		    kb_limit = kb2;
		  }else{
		    jr = 0;
		    jb_limit = jb;

		    ir_base = th_id * jr_chunk * MR;
		    kb_limit = min(th_id * jr_chunk * MR + jr_chunk * MR, kb2);
		  }
	          for (; jr < jb_limit; jr += NR) {
	            nr = min(jb-jr, NR);
	            jr2 = jr/NR;
	            for (ir=ir_base; ir < kb_limit; ir += MR) {
	  	      mr = min(kb_limit-ir, MR);
		      gemm_ukernel_asm(nr, mr, NR, MR, ib, &alpha, 
		 	               &FBrow_NHWC(j2 * Cob_Nr + jr2, i, n, m, 0), &Ac[ir * ib], 
		                       &beta, &Ctmp[omp_get_thread_num() * (MR * NR)], &Yrow_NHWC(h, j + jr, l, k + ir), ldY3);
	            }
                  }
                }
                #pragma omp barrier
              }
            } 
          } 
      } 
  }

}
*/

/*
void packRB_neon( char orderA, char transA, int mc, int nc, DTYPE *A, int ldA, DTYPE *Ac, int RR ) {
  
    int    i, j, ii, ii_tmp, k, rr, iters, iters_left;
    float32x4_t A00, A01, A10, A11, A20, A21, A30, A31;

    if (((transA == 'N') && (orderA == 'C')) ||
        ((transA == 'T') && (orderA == 'R'))) {
      printf("Packing not yet implemented\n");
      exit(-1);
    } else {

      for ( i=0; i < mc; i += RR ) { 
        k = i * nc;
        rr = min( mc - i, RR );
         
        for ( i=0; i<mc; i+=RR ) {
          k = i*nc;
          rr = min( mc-i, RR );
          
	  if (rr == RR) { //rr == 8

	    for ( j=0; j < nc - 4; j += 4 ) {

	      A00[0] = Acol(j + 0, i);
	      A10[0] = Acol(j + 1, i);
	      A20[0] = Acol(j + 2, i);
	      A30[0] = Acol(j + 3, i);
	      
	      A00[1] = Acol(j + 0, i + 1);
	      A10[1] = Acol(j + 1, i + 1);
	      A20[1] = Acol(j + 2, i + 1);
	      A30[1] = Acol(j + 3, i + 1);
	      
	      A00[2] = Acol(j + 0, i + 2);
	      A10[2] = Acol(j + 1, i + 2);
	      A20[2] = Acol(j + 2, i + 2);
	      A30[2] = Acol(j + 3, i + 2);
              
	      A00[3] = Acol(j + 0, i + 3);
	      A10[3] = Acol(j + 1, i + 3);
	      A20[3] = Acol(j + 2, i + 3);
	      A30[3] = Acol(j + 3, i + 3);

	      A01[0] = Acol(j + 0, i + 4);
	      A11[0] = Acol(j + 1, i + 4);
	      A21[0] = Acol(j + 2, i + 4);
	      A31[0] = Acol(j + 3, i + 4);
	      
	      A01[1] = Acol(j + 0, i + 5);
	      A11[1] = Acol(j + 1, i + 5);
	      A21[1] = Acol(j + 2, i + 5);
	      A31[1] = Acol(j + 3, i + 5);
	      
	      A01[2] = Acol(j + 0, i + 6);
	      A11[2] = Acol(j + 1, i + 6);
	      A21[2] = Acol(j + 2, i + 6);
	      A31[2] = Acol(j + 3, i + 6);
              
	      A01[3] = Acol(j + 0, i + 7);
	      A11[3] = Acol(j + 1, i + 7);
	      A21[3] = Acol(j + 2, i + 7);
	      A31[3] = Acol(j + 3, i + 7);

              vst1q_f32(&Ac[k], A00); k += 4;
              vst1q_f32(&Ac[k], A01); k += 4;
              k += (RR - rr);
	      
              vst1q_f32(&Ac[k], A10); k += 4;
              vst1q_f32(&Ac[k], A11); k += 4;
              k += (RR - rr);
              
	      vst1q_f32(&Ac[k], A20); k += 4;
              vst1q_f32(&Ac[k], A21); k += 4;
              k += (RR - rr);
              
	      vst1q_f32(&Ac[k], A30); k += 4;
              vst1q_f32(&Ac[k], A31); k += 4;
              k += (RR - rr);

	    }

	    for(; j < nc; j++) {
              A00[0] = Acol(j, i+0);
              A00[1] = Acol(j, i+1);
              A00[2] = Acol(j, i+2);
              A00[3] = Acol(j, i+3);

              A01[0] = Acol(j, i+4);
              A01[1] = Acol(j, i+5);
              A01[2] = Acol(j, i+6);
              A01[3] = Acol(j, i+7);

              vst1q_f32(&Ac[k], A00); k += 4;
              vst1q_f32(&Ac[k], A01); k += 4;

              k += (RR - rr);
	    }
	  } else if (rr == 4) {
	  
	    for ( j=0; j < nc - 4; j += 4 ) {

	      A00[0] = Acol(j + 0, i);
	      A10[0] = Acol(j + 1, i);
	      A20[0] = Acol(j + 2, i);
	      A30[0] = Acol(j + 3, i);
	      
	      A00[1] = Acol(j + 0, i + 1);
	      A10[1] = Acol(j + 1, i + 1);
	      A20[1] = Acol(j + 2, i + 1);
	      A30[1] = Acol(j + 3, i + 1);
	      
	      A00[2] = Acol(j + 0, i + 2);
	      A10[2] = Acol(j + 1, i + 2);
	      A20[2] = Acol(j + 2, i + 2);
	      A30[2] = Acol(j + 3, i + 2);
              
	      A00[3] = Acol(j + 0, i + 3);
	      A10[3] = Acol(j + 1, i + 3);
	      A20[3] = Acol(j + 2, i + 3);
	      A30[3] = Acol(j + 3, i + 3);

              vst1q_f32(&Ac[k], A00); k += 4;
              k += (RR - rr);
	      
              vst1q_f32(&Ac[k], A10); k += 4;
              k += (RR - rr);
              
	      vst1q_f32(&Ac[k], A20); k += 4;
              k += (RR - rr);
              
	      vst1q_f32(&Ac[k], A30); k += 4;
              k += (RR - rr);

	    }

	    for(; j < nc; j++) {
              A00[0] = Acol(j, i + 0);
              A00[1] = Acol(j, i + 1);
              A00[2] = Acol(j, i + 2);
              A00[3] = Acol(j, i + 3);
              
              vst1q_f32(&Ac[k], A00); k += 4; 
              k += (RR - rr);

	    }
	  
	  } else { // rr != 8 && rr != 4
	  
	    for ( j=0; j<nc; j++ ) {
              for ( ii=0; ii<rr; ii++ ) {
                Ac[k] = Acol(j,i+ii);
                k++;
              }
              k += (RR-rr);
            }
	  
	  }
        }

      }
    }

}
*/
