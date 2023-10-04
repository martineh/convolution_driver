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


void convDirect_original( int n, int k, int c, 
                          int h, int w, 
                          int ho, int wo, 
                          int r, int s, 
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
      x_x = ih + ir;
      if (0 <= x_x && x_x < h) 
	for ( is=0; is<s; is++ ) {
	  x_y = iw + is;
	  if (0 <= x_y && x_y < w) {
            //printf("FB %d %d %d %d %16.10e\n", ik, ic, ir, is, Frow_NHWC(ik,ic,ir,is));
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


void convDirect_renamed( int t, int Co, int Ci, 
                         int Ho, int Wo, 
                         int ho, int wo, 
                         int Hf, int Wf, 
                         DTYPE *D, int ldD1, int ldD2, int ldD3,
	                 DTYPE *F, int ldF1, int ldF2, int ldF3,
                         DTYPE *Y, int ldY1, int ldY2, int ldY3,
			 int tformat)
{ 
  int h, i, j, k, l, m, n, x_x, x_y;

  // Quick return if possible
  if ( (t==0)||(Co==0)||(Ci==0)||
       (Ho==0)||(Wo==0)||
       (Hf==0)||(Wf==0))
    return;

  if (tformat == NHWC) {
    for ( h=0;  h<t;   h++ ) 
    for ( i=0;  i<Ci;   i++ ) 
    for ( j=0;  j<Co;   j++ ) 
    for ( k=0;  k<wo;   k++ ) 
    for ( l=0;  l<ho;   l++ ) 
    for ( m=0;  m<Wf;   m++ ) {
      x_y = k + m;
      if (0 <= x_y && x_y < Wo) {
	for ( n=0;  n<Hf;   n++ ) {
	  x_x = l + n;
	  if (0 <= x_x && x_x < Ho)
	    Yrow_NHWC(h,j,l,k) += Drow_NHWC(h,i,x_x,x_y) * Frow_NHWC(j,i,n,m);
	}
      }
    }
  } else {
    for ( h=0;  h<t;   h++ ) 
    for ( i=0;  i<Ci;   i++ ) 
    for ( j=0;  j<Co;   j++ ) 
    for ( k=0;  k<wo;   k++ ) 
    for ( l=0;  l<ho;   l++ ) 
    for ( m=0;  m<Wf;   m++ ) {
      x_y = k + m;
      if (0 <= x_y && x_y < Wo) {
	for ( n=0;  n<Hf;   n++ ) {
	  x_x = l + n;
	  if (0 <= x_x && x_x < Ho)
	    Yrow_NCHW(h,j,l,k) += Drow_NCHW(h,i,x_x,x_y) * Frow_NCHW(j,i,n,m);
	}
      }
    }

  }
}



void convDirect_reorder( int t, int Co, int Ci, 
                         int Ho, int Wo, 
                         int ho, int wo, 
                         int Hf, int Wf, 
                         DTYPE *D, int ldD1, int ldD2, int ldD3,
	                 DTYPE *F, int ldF1, int ldF2, int ldF3,
                         DTYPE *Y, int ldY1, int ldY2, int ldY3,
			 int tformat)
{ 

  int h, i, j, k, l, m, n, x_x, x_y;

  // Quick return if possible
  if ( (t==0) ||(Co==0)||(Ci==0)||
       (Ho==0)||(Wo==0)||
       (Hf==0)||(Wf==0) )
    return;

  if (tformat == NHWC) {
    for ( h=0;  h<t; h++ )
      for ( l=0;  l<ho; l++ )
	for ( n=0;  n<Hf; n++ ) {
	  x_x = l + n;
	  if (0 <= x_x && x_x < Ho)
	    for ( m=0;  m<Wf; m++ )
	      for ( i=0;  i<Ci; i++ )
		for ( k=0;  k<wo; k++ ) {
		  x_y = k + m;
		  if (0 <= x_y && x_y < Wo)
		    for ( j=0;  j<Co;   j++ )
		      Yrow_NHWC(h,j,l,k) += Drow_NHWC(h,i,x_x,x_y) * Frow_NHWC(j,i,n,m);
		}
	}
  } else {
    for ( h=0;  h<t; h++ )
      for ( l=0;  l<ho; l++ )
	for ( n=0;  n<Hf; n++ ) {
	  x_x = l + n;
	  if (0 <= x_x && x_x < Ho)
	    for ( m=0;  m<Wf; m++ )
	      for ( i=0;  i<Ci; i++ )
		for ( k=0;  k<wo; k++ ) {
		  x_y = k + m;
		  if (0 <= x_y && x_y < Wo)
		    for ( j=0;  j<Co;   j++ )
		      Yrow_NCHW(h,j,l,k) += Drow_NCHW(h,i,x_x,x_y) * Frow_NCHW(j,i,n,m);
		}
	}
  }
  
}

void convDirect_block( int t,     int Co,   int Ci, 
                       int Ho,    int Wo, 
                       int ho,    int wo, 
                       int Hf,    int Wf,
		       DTYPE *D,  int ldD1, int ldD2, int ldD3,
	               DTYPE *F,  int ldF1, int ldF2, int ldF3,
                       DTYPE *Y, int ldY1, int ldY2, int ldY3,
		       int tformat, int CIB, int COB, int WOB)
{ 

  int h, i, j, k, l, m, n, x_x, x_y, 
      ii, jj, kk, ib, jb, kb;

  // Quick return if possible
  if ( (t==0) ||(Co==0)||(Ci==0)||
       (Ho==0)||(Wo==0)||
       (Hf==0)||(Wf==0) )
    return;

  if (tformat == NHWC) {
    for ( h=0; h<t; h++ ) 
      for ( j=0; j<Co; j+=COB ) {
	jb = min(Co-j, COB);
	for ( i=0; i<Ci; i+=CIB ) {
	  ib = min(Ci-i, CIB);
	  for ( l=0; l<ho; l++ ) 
	    for ( k=0; k<wo; k+=WOB ) {
	      kb = min(wo-k, WOB);
	      for ( n=0; n<Hf; n++ ) {
		x_x = l + n;
		if (0 <= x_x && x_x < Ho)
		  for ( m=0; m<Wf; m++ )
		    for ( ii=0; ii<ib; ii++ )
		      for ( kk=0; kk<kb; kk++ ) {
			x_y = k + kk + m;
			if (0 <= x_y && x_y < Wo)
			  for ( jj=0; jj<jb; jj++ )
			    Yrow_NHWC(h,j+jj,l,k+kk) += Drow_NHWC(h,i+ii,l+n,k+kk+m) * Frow_NHWC(j+jj,i+ii,n,m);
		      }
	      }
	    }
	}
      }
  } else {
    for ( h=0; h<t; h++ ) 
      for ( j=0; j<Co; j+=COB ) {
	jb = min(Co-j, COB);
	for ( i=0; i<Ci; i+=CIB ) {
	  ib = min(Ci-i, CIB);
	  for ( l=0; l<ho; l++ ) 
	    for ( k=0; k<wo; k+=WOB ) {
	      kb = min(wo-k, WOB);
	      for ( n=0; n<Hf; n++ ) {
		x_x = l + n;
		if (0 <= x_x && x_x < Ho)
		  for ( m=0; m<Wf; m++ )
		    for ( ii=0; ii<ib; ii++ )
		      for ( kk=0; kk<kb; kk++ ) {
			x_y = k + kk + m;
			if (0 <= x_y && x_y < Wo)
			  for ( jj=0; jj<jb; jj++ )
			    Yrow_NCHW(h,j+jj,l,k+kk) += Drow_NCHW(h,i+ii,l+n,k+kk+m) * Frow_NCHW(j+jj,i+ii,n,m);
		      }
	      }
	    }
	}
      }

  }
} 


void transform_input_tzemeng( int t, int Ci,
			      int Ho, int Wo,
			      int ho, int wo,
			      int Hf, int Wf,
			      DTYPE *D, int ldD1, int ldD2, int ldD3,
			      DTYPE *DT, int ldDT1, int ldDT2, int ldDT3, int ldDT4,
			      int tformat, int CIB) {
  int     h,
          i, j,
          k, l,
          m, n,
          ii, jj, kk,
          ib, jb, kb;

  int i2, x;

  if ( (t==0) ||(Ci==0)||
       (Ho==0)||(Wo==0)||
       (Hf==0)||(Wf==0) )
    return;

  if (tformat == NHWC) {
    for ( h=0; h<t; h++)
      for ( l=0; l<Ho; l++)
	for ( k=0; k<Wo; k++)
	  for ( i=0,i2=0; i<Ci; i+=CIB,i2++) {
	    ib = min(Ci-i, CIB);
	    for ( ii=0; ii<ib; ii++)
	      DT(h, i2, l, k, ii) = Drow_NHWC(h, i+ii, l, k);	  
	      }
  } else {
    for ( h=0; h<t; h++)
      for ( l=0; l<Ho; l++)
	for ( k=0; k<Wo; k++)
	  for ( i=0,i2=0; i<Ci; i+=CIB,i2++) {
	    ib = min(Ci-i, CIB);
	    for ( ii=0; ii<ib; ii++)
	      DT(h, i2, l, k, ii) = Drow_NCHW(h, i+ii, l, k);	  
	      }
    
  }

}

void transform_output_tzemeng( int t, int Co,
			       int Ho, int Wo,
			       int ho, int wo,
			       int Hf, int Wf,
			       DTYPE *Y, int ldY1, int ldY2, int ldY3,
			       DTYPE *YT, int ldYT1, int ldYT2, int ldYT3, int ldYT4,
			       int tformat, int COB) {
  int h,
      i, i2, j,
      k, l,
      m, n,
      ii, jj, kk,
      ib, jb, kb;

  // Quick return if possible
  if ( (t==0) ||(Co==0)||
       (Ho==0)||(Wo==0)||
       (Hf==0)||(Wf==0) )
    return;

  if (tformat == NHWC) {
    for ( h=0; h<t; h++)
      for ( l=0; l<ho; l++)
	for ( k=0; k<wo; k++)
	  for ( i=0,i2=0; i<Co; i+=COB,i2++) {
	    ib = min(Co-i, COB);
	    for ( ii=0; ii<ib; ii++)
	      Yrow_NHWC(h, i+ii, l, k) = YT(h, i2, l, k, ii);	  
	  }
  } else {
    for ( h=0; h<t; h++)
      for ( l=0; l<ho; l++)
	for ( k=0; k<wo; k++)
	  for ( i=0,i2=0; i<Co; i+=COB,i2++) {
	    ib = min(Co-i, COB);
	    for ( ii=0; ii<ib; ii++)
	      Yrow_NCHW(h, i+ii, l, k) = YT(h, i2, l, k, ii);	  
	  }
  }
}



void transform_filter_tzemeng( int Ci, int Co,
			       int Hf, int Wf,
			       DTYPE *F, int ldF1, int ldF2, int ldF3,
			       DTYPE *FT, int ldFT1, int ldFT2, int ldFT3, int ldFT4, int ldFT5,
			       int tformat, int CIB, int COB) {
  int     h,
          i, j,
          k, l,
          m, n,
          ii, jj, kk,
          ib, jb, kb;
  int i2, j2;

  // Quick return if possible
  if ( (Ci==0)||(Co==0)||
       (Hf==0)||(Wf==0) )
    return;

  if (tformat == NHWC) {
    for ( j=0,j2=0; j<Co; j+=COB,j2++ ) {
      jb = min(Co-j, COB);
      for ( i=0,i2=0; i<Ci; i+=CIB,i2++ ) {
	ib = min(Ci-i, CIB);       
	for ( n=0; n<Hf; n++ )
	  for ( m=0; m<Wf; m++ )
	    for ( ii=0; ii<ib; ii++ )
	      for ( jj=0; jj<jb; jj++ )
		FT(i2, j2, n, m, ii, jj ) = Frow_NHWC(j+jj, i+ii, n,  m);
      }
    }
  } else {
    for ( j=0,j2=0; j<Co; j+=COB,j2++ ) {
      jb = min(Co-j, COB);
      for ( i=0,i2=0; i<Ci; i+=CIB,i2++ ) {
	ib = min(Ci-i, CIB);       
	for ( n=0; n<Hf; n++ )
	  for ( m=0; m<Wf; m++ )
	    for ( ii=0; ii<ib; ii++ )
	      for ( jj=0; jj<jb; jj++ )
		FT(i2, j2, n, m, ii, jj ) = Frow_NCHW(j+jj, i+ii, n,  m);
      }
    }
  }
  
}

#if TH == 1
void convDirect_block_tzemeng( int t, int Co, int Ci,
			       int Ho, int Wo,
			       int ho, int wo,
			       int Hf, int Wf,
			       DTYPE *Ctmp,
			       DTYPE *DT, int ldDT1, int ldDT2, int ldDT3, int ldDT4,
			       DTYPE *FT, int ldFT1, int ldFT2, int ldFT3, int ldFT4, int ldFT5,
			       DTYPE *YT, int ldYT1, int ldYT2, int ldYT3, int ldYT4,
			       int tformat, int CIB, int COB, int WOB) {
  int  h,
    i, j, i2, j2,
    k, l,
    m, n,
    ii, jj, kk,
    ib, jb, kb, ob;

  int n_if  = 0;
  int n_else = 0;
  int o;
  
  float alpha = 1.0;
  float beta  = 1.0;
  
  // Quick return if possible
  if ( (t==0) ||(Co==0)||(Ci==0)||
       (Ho==0)||(Wo==0)||
       (Hf==0)||(Wf==0) )
    return;


  // Loops reordered as in "High Peformance Zero-Memory Overhead Direct Convolution" by J. Zhang et al, 2018
  for ( h=0; h<t; h++ ) {
    for ( j=0,j2=0; j<Co; j+=COB,j2++ ) {
      jb = min(Co-j, COB);
      for ( i=0,i2=0; i<Ci; i+=CIB,i2++ ) {
        ib = min(Ci-i, CIB);
        for ( l=0; l<ho; l++ ) {
          for ( k=0; k<wo; k+=WOB ) {
            kb = min(wo-k, WOB);
            for ( n=0; n<min(Hf,Ho-l); n++ ) {
              for ( m=0; m<Wf; m++ ) {
		 ob = min(kb,Wo-k-m+1);
		  if ((ob == MR) && (jb == NR))
		    gemm_microkernel_Cresident_neon_7x12_unroll_4_fp32_fixed( ob, jb, ib,
									1.0, &DT(h, i2, l+n, k+m, 0), 
									&FT(i2, j2, n, m, 0, 0), 
									1.0, &YT(h, j2, l, k, 0), ldYT4 );
                  else
		    gemm_microkernel_Cresident_neon_7x12_unroll_4_fp32( ob, jb, ib,
									1.0, &DT(h, i2, l+n, k+m, 0), 
									&FT(i2, j2, n, m, 0, 0), 
									1.0, &YT(h, j2, l, k, 0), ldYT4 );

	      } } } } } } }
  
}
#else
void convDirect_block_tzemeng( int t, int Co, int Ci,
			       int Ho, int Wo,
			       int ho, int wo,
			       int Hf, int Wf,
			       DTYPE *Ctmp,
			       DTYPE *DT, int ldDT1, int ldDT2, int ldDT3, int ldDT4,
			       DTYPE *FT, int ldFT1, int ldFT2, int ldFT3, int ldFT4, int ldFT5,
			       DTYPE *YT, int ldYT1, int ldYT2, int ldYT3, int ldYT4,
			       int tformat, int CIB, int COB, int WOB) {
  int  h,
    i, j, i2, j2,
    k, l,
    m, n,
    ii, jj, kk,
    ib, jb, kb, ob;

  int n_if  = 0;
  int n_else = 0;
  int o;
  
  float alpha = 1.0;
  float beta  = 1.0;
  
  // Quick return if possible
  if ( (t==0) ||(Co==0)||(Ci==0)||
       (Ho==0)||(Wo==0)||
       (Hf==0)||(Wf==0) )
    return;

  int ho_chunk = (int) ceil((double)ho/TH); 

  #pragma omp parallel num_threads(TH) private(h, j, j2, jb, i, i2, ib, l, k, kb, n, m, ob) firstprivate(ho_chunk)
  {
  int th_id = omp_get_thread_num();
  for ( h=0; h<t; h++ ) {
    for ( j=0,j2=0; j<Co; j+=COB,j2++ ) {
      jb = min(Co-j, COB);
      for ( i=0,i2=0; i<Ci; i+=CIB,i2++ ) {
        ib = min(Ci-i, CIB);
        for ( l=ho_chunk*th_id; l< min(ho_chunk*th_id + ho_chunk, ho); l++ ) 
        //for ( l=0; l<ho; l++ ) {
          for ( k=0; k<wo; k+=WOB ) {
            kb = min(wo-k, WOB);
            for ( n=0; n<min(Hf,Ho-l); n++ ) {
              for ( m=0; m<Wf; m++ ) {
		 ob = min(kb,Wo-k-m+1);
		  if ((ob == MR) && (jb == NR))
		    gemm_microkernel_Cresident_neon_7x12_unroll_4_fp32_fixed( ob, jb, ib,
									1.0, &DT(h, i2, l+n, k+m, 0), 
									&FT(i2, j2, n, m, 0, 0), 
									1.0, &YT(h, j2, l, k, 0), ldYT4 );
                  else
		    gemm_microkernel_Cresident_neon_7x12_unroll_4_fp32( ob, jb, ib,
									1.0, &DT(h, i2, l+n, k+m, 0), 
									&FT(i2, j2, n, m, 0, 0), 
									1.0, &YT(h, j2, l, k, 0), ldYT4 );

	      } } } } } } }
  
}
#endif

void transform_filter_block_shalom( int Ci, int Co,
				    int Hf, int Wf,
				    DTYPE *F,  int ldF1,  int ldF2,  int ldF3,
				    DTYPE *FB, int ldFB1, int ldFB2, int ldFB3, int ldFB4,
				    int tformat) {
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



#if TH == 1
void convDirect_block_shalom( int t,     int Co,   int Ci, 
			      int Ho,    int Wo, 
			      int ho,    int wo, 
			      int Hf,    int Wf,
			      DTYPE *D,  int ldD1,  int ldD2,  int ldD3,
			      DTYPE *FB, int ldFB1, int ldFB2, int ldFB3, int ldFB4,
			      DTYPE *Y,  int ldY1,  int ldY2,  int ldY3,
			      int tformat, int CIB, int COB, int WOB) { 

  int h, i, j, k, l, m, n, i2, j2,
      ii, jj, kk, ib, jb, kb, Cob_Nr = COB/NR;

  int jr, nr, jr2, ir, mr;

  // Quick return if possible
  if ( (t==0) ||(Co==0)||(Ci==0)||
       (Ho==0)||(Wo==0)||
       (Hf==0)||(Wf==0) )
    return;

   
  if (tformat == NHWC) { 
     for ( h=0; h<t; h++ ) 
       for ( j=0,j2=0; j<Co; j+=COB,j2++ ) { 
         jb = min(Co-j, COB); 
         for ( i=0,i2=0; i<Ci; i+=CIB,i2++ ) { 
           ib = min(Ci-i, CIB); 
           for ( l=0; l<ho; l++ ) 
             for ( k=0; k<wo; k+=WOB ) { 
               kb = min(wo-k, WOB); 
               for ( n=0; n<min(Hf,Ho-l); n++ )
                 for ( m=0; m<Wf; m++ ) 
		  for ( jr=0, jr2=0; jr < jb; jr += NR, jr2++) {
		    nr = min(jb-jr, NR);
		    for ( ir=0; ir < min(kb, Wo-k-m+1); ir += MR) {
		      mr = min(min(kb, Wo-k-m+1)-ir, MR);
                      #if (MR==7) && (NR==12) 
		       if ((mr == MR) && (nr == NR))
			 gemm_microkernel_Cresident_neon_7x12_nopackA_unroll_4_fp32_fixed( mr, nr, ib, 
		                                                                          1.0, &Drow_NHWC(h, i, l+n, k+ir+m), ldD3,//4
											  &FBrow_NHWC(j2*Cob_Nr+jr2, i, n, m, 0),  
											  1.0, &Yrow_NHWC(h, j+jr, l, k+ir), ldY3 );
		      else
			gemm_microkernel_Cresident_neon_7x12_nopackA_unroll_4_fp32( mr, nr, ib, 
										    1.0, &Drow_NHWC(h, i, l+n, k+ir+m), ldD3,//4
										    &FBrow_NHWC(j2*Cob_Nr+jr2, i, n, m, 0),  
										    1.0, &Yrow_NHWC(h, j+jr, l, k+ir), ldY3 );
                     #else
		       printf("ERROR: Microkernel doesn't exist.\n");
		       exit(-1);
                     #endif
		    }
                  }
             } 
         } 
       } 
   } else { 
     printf("1. Case not yet implemented %d\n", tformat); 
     exit(-1); 
   } 
}
#else
void convDirect_block_shalom( int t,     int Co,   int Ci, 
			      int Ho,    int Wo, 
			      int ho,    int wo, 
			      int Hf,    int Wf,
			      DTYPE *D,  int ldD1,  int ldD2,  int ldD3,
			      DTYPE *FB, int ldFB1, int ldFB2, int ldFB3, int ldFB4,
			      DTYPE *Y,  int ldY1,  int ldY2,  int ldY3,
			      int tformat, int CIB, int COB, int WOB) { 

  int h, i, j, k, l, m, n, i2, j2,
      ii, jj, kk, ib, jb, kb, Cob_Nr = COB/NR;

  int jr, nr, jr2, ir, mr;

  // Quick return if possible
  if ( (t==0) ||(Co==0)||(Ci==0)||
       (Ho==0)||(Wo==0)||
       (Hf==0)||(Wf==0) )
    return;

  int ho_chunk = (int) ceil((double)ho/TH); 
  int kb_limit; 
  if (tformat == NHWC) { 
    #pragma omp parallel num_threads(TH) private(h, j, j2, jb, i, i2, ib, l, k, kb, n, m, jr, nr, jr2, ir, mr, kb_limit) firstprivate(ho_chunk)
    {
     int th_id = omp_get_thread_num();
     for ( h=0; h<t; h++ ) 
       for ( j=0,j2=0; j<Co; j+=COB,j2++ ) { 
         jb = min(Co-j, COB); 
         for ( i=0,i2=0; i<Ci; i+=CIB,i2++ ) { 
           ib = min(Ci-i, CIB); 
           for ( l=ho_chunk*th_id; l< min(ho_chunk*th_id + ho_chunk, ho); l++ ) 
           //for ( l=0; l<ho; l++ ) 
             for ( k=0; k<wo; k+=WOB ) { 
               kb = min(wo-k, WOB); 
               for ( n=0; n<min(Hf,Ho-l); n++ )
                 for ( m=0; m<Wf; m++ ) 
		  for ( jr=0, jr2=0; jr < jb; jr += NR, jr2++) {
		    nr = min(jb-jr, NR);
	            kb_limit = min(kb, Wo-k-m+1);
		    for ( ir=0; ir < kb_limit; ir += MR) {
		      mr = min(kb_limit-ir, MR);
                      #if (MR==7) && (NR==12) 
		       if ((mr == MR) && (nr == NR))
			 gemm_microkernel_Cresident_neon_7x12_nopackA_unroll_4_fp32_fixed( mr, nr, ib, 
		                                                                          1.0, &Drow_NHWC(h, i, l+n, k+ir+m), ldD3,//4
											  &FBrow_NHWC(j2*Cob_Nr+jr2, i, n, m, 0),  
											  1.0, &Yrow_NHWC(h, j+jr, l, k+ir), ldY3 );
		      else
			gemm_microkernel_Cresident_neon_7x12_nopackA_unroll_4_fp32( mr, nr, ib, 
										    1.0, &Drow_NHWC(h, i, l+n, k+ir+m), ldD3,//4
										    &FBrow_NHWC(j2*Cob_Nr+jr2, i, n, m, 0),  
										    1.0, &Yrow_NHWC(h, j+jr, l, k+ir), ldY3 );
                     #else
		       printf("ERROR: Microkernel doesn't exist.\n");
		       exit(-1);
                     #endif
		    }
                  }
             } 
         } 
       } 
    }
   } else { 
     printf("1. Case not yet implemented %d\n", tformat); 
     exit(-1); 
   } 
}
#endif

void transform_filter_block_blis( int Ci, int Co,
				  int Hf, int Wf,
				  DTYPE *F,  int ldF1,  int ldF2,  int ldF3,
				  DTYPE *FB, int ldFB1, int ldFB2, int ldFB3, int ldFB4,
				  int tformat) {
  int  i, j, jj, jb, j2, m, n;  
  // Quick return if possible
  if ( (Ci==0)||(Co==0)||
       (Hf==0)||(Wf==0) )
    return;

  if (tformat == NHWC) {
#ifdef MK_BLIS
    /* Prepare to call micro-kernel with transposed operands */
    for ( j=0,j2=0; j<Co; j+=MR,j2++ ) {
      jb = min(Co-j, MR);
#else
    for ( j=0,j2=0; j<Co; j+=NR,j2++ ) {
      jb = min(Co-j, NR);
#endif
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

#if TH == 1
void convDirect_block_blis( int t,     int Co,   int Ci, 
                            int Ho,    int Wo, 
                            int ho,    int wo, 
                            int Hf,    int Wf,
		            DTYPE *D,  int ldD1,  int ldD2,  int ldD3,
	                    DTYPE *FB, int ldFB1, int ldFB2, int ldFB3, int ldFB4,
                            DTYPE *Y,  int ldY1,  int ldY2,  int ldY3,
                            DTYPE *Ac, DTYPE *Ctmp,
		            int tformat, int CIB, int COB, int WOB) { 

  int blis_mr, blis_nr;
  
    blis_mr = MR;
    blis_nr = NR;
  
  int h, i, j, k, l, m, n, i2, j2,
      ii, jj, kk, ib, jb, kb, Cob_Nr = COB/NR, Cob_Mr = COB/MR;

  int jr, nr, jr2, ir, mr, in = 0;

  float alpha = 1.0;
  float beta  = 1.0;
  // Quick return if possible
  if ( (t==0) ||(Co==0)||(Ci==0)||
       (Ho==0)||(Wo==0)||
       (Hf==0)||(Wf==0) )
    return;

  if (tformat == NHWC) { 
     for ( h=0; h<t; h++ ) 
         for ( i=0,i2=0; i<Ci; i+=CIB,i2++ ) { 
           ib = min(Ci-i, CIB); 
           for ( l=0; l<ho; l++ ) {
             for ( k=0; k<wo; k+=WOB ) { 
               kb = min(wo-k, WOB); 
               for ( n=0; n<min(Hf,Ho-l); n++ ) {
                 for ( m=0; m<Wf; m++ ) {
                   packRB( 'R', 'N', kb, ib, &Drow_NHWC(h, i, l+n, k+m), ldD3, Ac, blis_mr);
		   //packRB_neon('R', 'N', kb, ib, &Drow_NHWC(h, i, l + n, k + m), ldD3, Ac, MR);
		   for ( j=0,j2=0; j<Co; j+=COB,j2++ ) { 
		     jb = min(Co-j, COB); 
		     for ( jr=0, jr2=0; jr < jb; jr += blis_nr, jr2++) {
		       nr = min(jb-jr, blis_nr);
		       for ( ir=0; ir < min(kb, Wo-k-m+1); ir += blis_mr) {
			 mr = min(min(kb, Wo-k-m+1)-ir, blis_mr);
			 gemm_ukernel_asm(nr, mr, NR, MR, ib, &alpha, 
			 	         &FBrow_NHWC(j2 * Cob_Nr + jr2, i, n, m, 0), &Ac[ir*ib], 
		                         &beta, Ctmp, &Yrow_NHWC(h, j + jr, l, k + ir), ldY3);
		       }
                     }
                   }
                 }
	       }
	     } 
           } 
         } 
   } else { 
     printf("1. Case not yet implemented %d\n", tformat); 
     exit(-1); 
   }

}

#else
//ORIGINAL PARALLELIZATION
/*
void convDirect_block_blis( int t,     int Co,   int Ci, 
                            int Ho,    int Wo, 
                            int ho,    int wo, 
                            int Hf,    int Wf,
		            DTYPE *D,  int ldD1,  int ldD2,  int ldD3,
	                    DTYPE *FB, int ldFB1, int ldFB2, int ldFB3, int ldFB4,
                            DTYPE *Y,  int ldY1,  int ldY2,  int ldY3,
                            DTYPE *Ac, DTYPE *Ctmp,
		            int tformat, int CIB, int COB, int WOB) { 

  int h, i, j, k, l, m, n, i2, j2,
      ii, jj, kk, ib, jb, kb, Cob_Nr = COB/NR, Cob_Mr = COB/MR;
      //DTYPE Cc[MR*NR], blis_beta = 0.0;

  int jr, nr, jr2, ir, mr, in = 0;

  float alpha = 1.0;
  float beta  = 1.0;
  // Quick return if possible
  if ( (t==0) ||(Co==0)||(Ci==0)||
       (Ho==0)||(Wo==0)||
       (Hf==0)||(Wf==0) )
    return;

  if (tformat == NHWC) { 
     for ( h=0; h<t; h++ ) 
         for ( i=0,i2=0; i<Ci; i+=CIB,i2++ ) { 
           ib = min(Ci-i, CIB); 
           for ( l=0; l<ho; l++ ) 
             for ( k=0; k<wo; k+=WOB ) { 
               kb = min(wo-k, WOB); 
               for ( n=0; n<min(Hf,Ho-l); n++ )
                 for ( m=0; m<Wf; m++ ) {
                   packRB( 'R', 'N', kb, ib, &Drow_NHWC(h, i, l+n, k+m), ldD3, Ac, MR);
		   //packRB_neon('R', 'N', kb, ib, &Drow_NHWC(h, i, l + n, k + m), ldD3, Ac, MR);
		   for ( j=0,j2=0; j<Co; j+=COB,j2++ ) { 
		     jb = min(Co-j, COB);
                     #pragma omp parallel for private(nr, jr2, ir, mr)
		     for (jr = 0; jr < jb; jr += NR) {
		       nr = min(jb-jr, NR);
		       jr2 = jr/NR;
		       for ( ir=0; ir < min(kb, Wo-k-m+1); ir += MR) {
			 mr = min(min(kb, Wo-k-m+1)-ir, MR);
			 gemm_ukernel_asm(nr, mr, NR, MR, ib, &alpha, 
			 	          &FBrow_NHWC(j2 * Cob_Nr + jr2, i, n, m, 0), &Ac[ir*ib], 
		                          &beta, &Ctmp[omp_get_thread_num() * (MR * NR)], &Yrow_NHWC(h, j + jr, l, k + ir), ldY3);

		    }
                  }
                }
             } 
         } 
       } 
   } else { 
     printf("1. Case not yet implemented %d\n", tformat); 
     exit(-1); 
   }

}
*/

//NEW PARALLELIZATION OVER Ho loop

void convDirect_block_blis( int t,     int Co,   int Ci, 
                            int Ho,    int Wo, 
                            int ho,    int wo, 
                            int Hf,    int Wf,
		            DTYPE *D,  int ldD1,  int ldD2,  int ldD3,
	                    DTYPE *FB, int ldFB1, int ldFB2, int ldFB3, int ldFB4,
                            DTYPE *Y,  int ldY1,  int ldY2,  int ldY3,
                            DTYPE *Ac, DTYPE *Ctmp,
		            int tformat, int CIB, int COB, int WOB) { 

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
  int kb_limit;
  float alpha = 1.0;
  float beta  = 1.0;

  int ho_chunk = (int) ceil((double)ho/TH); 

  #pragma omp parallel num_threads(TH) private(h, i, i2, ib, l, k, kb, n, m, j, j2, jb, jr, nr, jr2, ir, mr, kb_limit) firstprivate(ho_chunk)
  {
    int th_id = omp_get_thread_num();
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
		      gemm_ukernel_asm(nr, mr, NR, MR, ib, &alpha, 
		 	               &FBrow_NHWC(j2 * Cob_Nr + jr2, i, n, m, 0), &Ac[(th_id * CIB * WOB) + (ir * ib)], 
		                       &beta, &Ctmp[omp_get_thread_num() * (MR * NR)], &Yrow_NHWC(h, j + jr, l, k + ir), ldY3);
	            }
                  }
                }
              }
            } 
          } 
        } 
    }
  }

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
		            int tformat, int CIB, int COB, int WOB) { 

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
#endif

