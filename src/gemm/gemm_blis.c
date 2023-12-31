/* 
   GEMM FLAVOURS

   -----

   GEMM FLAVOURS is a family of algorithms for matrix multiplication based
   on the BLIS approach for this operation: https://github.com/flame/blis

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

   author    = "Enrique S. Quintana-Orti"
   contact   = "quintana@disca.upv.es"
   copyright = "Copyright 2021, Universitat Politecnica de Valencia"
   license   = "GPLv3"
   status    = "Production"
   version   = "1.1"
*/

#include "gemm_blis.h"


int print_matrix(char *, char, size_t, int, DTYPE *, size_t);

//----------------------------------------------------------
//Loop jr (L4) Parallelization
//----------------------------------------------------------
void gemm_blis_B3A2C0( char orderA, char orderB, char orderC,
                       char transA, char transB, 
                       size_t m, size_t n, size_t k, 
                       DTYPE alpha, DTYPE *A, size_t ldA, 
		                    DTYPE *B, size_t ldB, 
		       DTYPE beta,  DTYPE *C, size_t ldC, 
		       DTYPE *Ac, DTYPE *Bc, 
                       size_t MC, size_t NC, size_t KC, 
		       int MR, int NR, int TH, int loop, DTYPE *Ctmp,
		       ukernel_asm ukr, ukernel_edge ukr_edge) {

  int    ic, jc, pc, mc, nc, kc, ir, jr, mr, nr, j, i; 
  DTYPE  zero = 0.0, one = 1.0, betaI; 
  DTYPE  *Aptr, *Bptr, *Cptr;
  float  beta_edge = 0.0;

  DTYPE *Ctmp_th;

  #if defined(CHECK)
  #include "check_params.h"
  #endif

  // Quick return if possible
  if ( (m==0)||(n==0)||(((alpha==zero)||(k==0))&&(beta==one)) )
    return;

  #include "quick_gemm.h"

  if (TH == 1) {
    for ( jc=0; jc<n; jc+=NC ) {
      nc = min(n-jc, NC); 
      for ( pc=0; pc<k; pc+=KC ) {
        kc = min(k-pc, KC); 
        Bptr = &Bcol(pc,jc);
        
	pack_CB( orderB, transB, kc, nc, Bptr, ldB, Bc, NR);
        
	if ( pc==0 ) betaI = beta;
        else betaI = one;
        
	for ( ic=0; ic<m; ic+=MC ) {
          mc = min(m-ic, MC); 
          Aptr = &Acol(ic, pc);
          pack_RB( orderA, transA, mc, kc, Aptr, ldA, Ac, MR);
          
          for (jr=0; jr<nc; jr+=NR ) {
            nr = min(nc-jr, NR); 
            for (ir=0; ir<mc; ir+=MR ) {
              mr = min(mc-ir, MR); 
              Cptr = &Ccol(ic+ir,jc+jr);
	      
  	      if (mr == MR && nr == NR)
                ukr(kc, &alpha, &Ac[ir*kc], &Bc[jr*kc], &betaI, Cptr, ldC * sizeof(float));
              else
	        ukr_edge(mr, nr, MR, NR, kc, &alpha, &Ac[ir*kc], &Bc[jr*kc], &betaI, Ctmp, Cptr, ldC);

            }
          }

        }
      }
    }
  
  } else {
    int th_id;
    DTYPE *Acptr;

    #ifdef OMP_ENABLE
    #pragma omp parallel num_threads(TH) private(jc, nc, pc, kc, Bptr, ic, mc, Aptr, Acptr, Cptr, Ctmp_th, jr, nr, ir, mr, th_id)
    {
      th_id = omp_get_thread_num();
    #else
      printf("ERROR: Parallel option configured but not compiled\n"); 
      exit(-1);
    #endif

    if(loop == 3) {
    //=========================================== LOOP 3 PARALLELIZATION =======================================//
    for ( jc=0; jc<n; jc+=NC ) {
      nc = min(n-jc, NC);
      int its_nc = (int) ceil((double)nc/NR/omp_get_num_threads());
      for ( pc=0; pc<k; pc+=KC ) {
        kc = min(k-pc, KC);
        Bptr = &Bcol(pc,jc+its_nc*NR*omp_get_thread_num());

        pack_CB( orderB, transB, kc, min(its_nc*NR, nc - its_nc*NR*omp_get_thread_num()), Bptr, ldB, Bc+kc*its_nc*NR*omp_get_thread_num(), NR);

        if ( pc==0 )  betaI = beta;
        else betaI = one;

        int its_m = (int) ceil((double)m/MC/omp_get_num_threads());
        Acptr = Ac + ((MC + MR )* (KC + 1)) * omp_get_thread_num();

        #ifdef OMP_ENABLE
          #pragma omp barrier
        #endif

        for ( ic=omp_get_thread_num()*(its_m * MC); ic<min(m, (omp_get_thread_num()+1) * (its_m * MC)); ic+=MC ) {
          mc = min(m-ic, MC);
          Aptr = &Acol(ic, pc);

          pack_RB( orderA, transA, mc, kc, Aptr, ldA, Acptr, MR);

          for ( jr=0; jr<nc; jr+=NR ) {
            nr = min(nc-jr, NR);
            for ( ir=0; ir<mc; ir+=MR ) {
              mr = min(mc-ir, MR);
              Cptr = &Ccol(ic+ir,jc+jr);

  	      if (mr == MR && nr == NR)
                ukr(kc, &alpha, &Acptr[ir*kc], &Bc[jr*kc], &betaI, Cptr, ldC * sizeof(float));
              else {
	        Ctmp_th = &Ctmp[th_id * MR * NR];
	        ukr_edge(mr, nr, MR, NR, kc, &alpha, &Acptr[ir*kc], &Bc[jr*kc], &betaI, Ctmp_th, Cptr, ldC);
	      }
	    }
          }
        }
        #ifdef OMP_ENABLE
          #pragma omp barrier
        #endif
      }
    }
    
    } else if(loop == 4) {

    //=========================================== LOOP 4 PARALLELIZATION =======================================//
    for ( jc=0; jc<n; jc+=NC ) {
      nc = min(n-jc, NC); 
      int its_nc = (int) ceil((double)nc/NR/TH);
      for ( pc=0; pc<k; pc+=KC ) {
        kc = min(k-pc, KC); 
        Bptr = &Bcol(pc, jc + its_nc * NR * th_id);
        pack_CB( orderB, transB, kc, min(its_nc * NR, nc - its_nc * NR * th_id), 
                 Bptr, ldB, Bc + kc * its_nc * NR * th_id, NR);
          
        if ( pc==0 ) betaI = beta;
        else betaI = one;

        for ( ic=0; ic<m; ic+=MC ) {
          mc = min(m-ic, MC); 

          int its_mc = (int) ceil((double)mc/MR/TH);
          Aptr = &Acol(ic + its_mc * MR * th_id,pc);

          pack_RB( orderA, transA, min(its_mc * MR, mc - its_mc * MR * th_id), kc, 
	           Aptr, ldA, Ac + kc * its_mc * MR * th_id, MR);

          #ifdef OMP_ENABLE
	  #pragma omp barrier
          #endif
          for ( jr=th_id*(its_nc*NR); jr<min(nc,(th_id+1)*(its_nc*NR)); jr+=NR ) {
            nr = min(nc-jr, NR); 

            for ( ir=0; ir<mc; ir+=MR ) {
              mr = min(mc-ir, MR); 
              Cptr = &Ccol(ic+ir,jc+jr);

  	      if (mr == MR && nr == NR)
                ukr(kc, &alpha, &Ac[ir*kc], &Bc[jr*kc], &betaI, Cptr, ldC * sizeof(float));
              else {
	        Ctmp_th = &Ctmp[th_id * MR * NR];
	        ukr_edge(mr, nr, MR, NR, kc, &alpha, &Ac[ir*kc], &Bc[jr*kc], &betaI, Ctmp_th, Cptr, ldC);
	      }

            }
          }
          #ifdef OMP_ENABLE
	  #pragma omp barrier
          #endif
        }
      }
    }
    
    } else if(loop == 5) {
     
    //=========================================== LOOP 5 PARALLELIZATION =======================================//
    for ( jc=0; jc<n; jc+=NC ) {
      nc = min(n-jc, NC); 
      int its_nc = (int) ceil((double)nc/NR/TH);

      for ( pc=0; pc<k; pc+=KC ) {
        kc = min(k-pc, KC); 
        Bptr = &Bcol(pc, jc + its_nc * NR * th_id);

        pack_CB( orderB, transB, kc, min(its_nc * NR, nc - its_nc * NR * th_id), 
                 Bptr, ldB, Bc + kc * its_nc * NR * th_id, NR);
          
	if ( pc==0 ) betaI = beta;
        else betaI = one;

        for ( ic=0; ic<m; ic+=MC ) {
          mc = min(m-ic, MC); 

	  int its_mc = (int) ceil((double)mc/MR/TH);
          Aptr = &Acol(ic + its_mc * MR * th_id,pc);

          pack_RB( orderA, transA, min(its_mc * MR, mc - its_mc * MR * th_id), kc, 
		   Aptr, ldA, Ac + kc * its_mc * MR * th_id, MR);

          #ifdef OMP_ENABLE
	  #pragma omp barrier
          #endif
          for ( jr=0; jr<nc; jr+=NR ) {
            nr = min(nc-jr, NR); 
            for ( ir=th_id*(its_mc*MR); ir<min(mc,(th_id+1)*(its_mc*MR)); ir+=MR ) {
              mr = min(mc-ir, MR); 
              Cptr = &Ccol(ic+ir,jc+jr);

  	      if (mr == MR && nr == NR)
                ukr(kc, &alpha, &Ac[ir*kc], &Bc[jr*kc], &betaI, Cptr, ldC * sizeof(float));
              else {
	        Ctmp_th = &Ctmp[th_id * MR * NR];
	        ukr_edge(mr, nr, MR, NR, kc, &alpha, &Ac[ir*kc], &Bc[jr*kc], &betaI, Ctmp_th, Cptr, ldC);
	      }

            }
          }
          #ifdef OMP_ENABLE
	  #pragma omp barrier
          #endif
        }
      }
    }
    //==========================================================================================================//
    }

    #ifdef OMP_ENABLE
    }
    #endif
  }
}

void gemm_blis_A3B2C0( char orderA, char orderB, char orderC,
                       char transA, char transB, 
                       size_t m, size_t n, size_t k, 
                       DTYPE alpha, DTYPE *A, size_t ldA, 
		                    DTYPE *B, size_t ldB, 
		       DTYPE beta,  DTYPE *C, size_t ldC, 
		       DTYPE *Ac, DTYPE *Bc, 
                       size_t MC, size_t NC, size_t KC, 
		       int MR, int NR, int TH, int loop, DTYPE *Ctmp,
		       ukernel_asm ukr, ukernel_edge ukr_edge) {

  int    ic, jc, pc, mc, nc, kc, ir, jr, mr, nr;
  DTYPE  zero = 0.0, one = 1.0, betaI;
  DTYPE  *Aptr, *Bptr, *Cptr, *Bcptr;
  DTYPE *Ctmp_th;

  #if defined(CHECK)
  #include "check_params.h"
  #endif

  // Quick return if possible
  if ( (m==0)||(n==0)||(((alpha==zero)||(k==0))&&(beta==one)) )
    return;

  #include "quick_gemm.h"

  if (TH == 1) {
    for ( ic=0; ic<m; ic+=MC ) {
      mc = min(m-ic, MC);

      for ( pc=0; pc<k; pc+=KC ) {
        kc = min(k-pc, KC);
        Aptr = &Acol(ic,pc);

        pack_RB( orderA, transA, mc, kc, Aptr, ldA, Ac, MR);

        if ( pc==0 ) betaI = beta;
        else betaI = one;

        for ( jc=0; jc<n; jc+=NC ) {
          nc = min(n-jc, NC);
          Bptr = &Bcol(pc,jc);

          pack_CB( orderB, transB, kc, nc, Bptr, ldB, Bc, NR);

          for ( ir=0; ir<mc; ir+=MR ) {
            mr = min(mc-ir, MR);
            for ( jr=0; jr<nc; jr+=NR ) {
              nr = min(nc-jr, NR);
              Cptr = &Ccol(ic+ir,jc+jr);
	      
  	      if (mr == MR && nr == NR)
                ukr(kc, &alpha, &Ac[ir*kc], &Bc[jr*kc], &betaI, Cptr, ldC * sizeof(float));
              else
	        ukr_edge(mr, nr, MR, NR, kc, &alpha, &Ac[ir*kc], &Bc[jr*kc], &betaI, Ctmp, Cptr, ldC);

	    }
          }
        }
      }
    }
  } else {
    int th_id; 
    #ifdef OMP_ENABLE
    #pragma omp parallel num_threads(TH) private(jc, nc, pc, kc, Bptr, Bcptr, ic, mc, Aptr, Cptr, Ctmp_th, jr, nr, ir, mr, th_id)
    {
      th_id = omp_get_thread_num();
    #else
      printf("ERROR: Parallel option configured but not compiled\n"); 
      exit(-1);
    #endif
    
    if(loop == 3) {
    //=========================================== LOOP 3 PARALLELIZATION =======================================//
    for ( ic=0; ic<m; ic+=MC ) {
      mc = min(m-ic, MC);
      int its_mc = (int) ceil((double)mc/MR/omp_get_num_threads());

      for ( pc=0; pc<k; pc+=KC ) {
        kc = min(k-pc, KC);
        Aptr = &Acol(ic+its_mc*MR*th_id, pc);
      
        pack_RB( orderA, transA, min(its_mc*MR, mc - its_mc*MR*th_id), 
		 kc, Aptr, ldA, Ac + kc*its_mc*MR*th_id, MR);

        if ( pc==0 ) betaI = beta;
        else betaI = one;

        int its_n = (int) ceil((double)n/NC/omp_get_num_threads());
        Bcptr = Bc + ((NC + NR )* (KC + 1)) * th_id;

        #ifdef OMP_ENABLE
        #pragma omp barrier
        #endif
        for ( jc=omp_get_thread_num()*(its_n * NC); jc<min(n, (th_id+1) * (its_n * NC)); jc+=NC ) {
          nc = min(n-jc, NC);
          Bptr = &Bcol(pc,jc);

          pack_CB( orderB, transB, kc, nc, Bptr, ldB, Bcptr, NR);

          for ( ir=0; ir<mc; ir+=MR ) {
            mr = min(mc-ir, MR);

            for ( jr=0; jr<nc; jr+=NR ) {
              nr = min(nc-jr, NR);
              Cptr = &Ccol(ic+ir,jc+jr);

  	      if (mr == MR && nr == NR)
                ukr(kc, &alpha, &Ac[ir*kc], &Bcptr[jr*kc], &betaI, Cptr, ldC * sizeof(float));
              else {
	        Ctmp_th = &Ctmp[th_id * MR * NR];
	        ukr_edge(mr, nr, MR, NR, kc, &alpha, &Ac[ir*kc], &Bcptr[jr*kc], &betaI, Ctmp_th, Cptr, ldC);
	      }

            }
          }
        }
        #ifdef OMP_ENABLE
        #pragma omp barrier
        #endif
      }
    }
    
    } else if(loop == 4) {
    
    //=========================================== LOOP 4 PARALLELIZATION =======================================//
    for ( ic=0; ic<m; ic+=MC ) {
      mc = min(m-ic, MC);
      int its_mc = (int) ceil((double)mc/MR/omp_get_num_threads());

      for ( pc=0; pc<k; pc+=KC ) {
        kc = min(k-pc, KC);
        Aptr = &Acol(ic+its_mc*MR*th_id, pc);
      
        pack_RB( orderA, transA, min(its_mc*MR, mc - its_mc*MR*th_id), 
		 kc, Aptr, ldA, Ac + kc*its_mc*MR*th_id, MR);

        if ( pc==0 ) betaI = beta;
        else betaI = one;

	for ( jc=0; jc<n; jc+=NC ) {
          nc = min(n-jc, NC);
          int its_nc = (int) ceil((double)nc/NR/TH);

          Bptr = &Bcol(pc, jc + its_nc * NR * th_id);
          pack_CB( orderB, transB, kc, min(its_nc * NR, nc - its_nc * NR * th_id), 
                   Bptr, ldB, Bc + kc * its_nc * NR * th_id, NR);

        #ifdef OMP_ENABLE
        #pragma omp barrier
        #endif
          for ( ir=th_id*(its_mc*MR); ir<min(mc,(th_id+1)*(its_mc*MR)); ir+=MR ) {
            mr = min(mc-ir, MR);

            for ( jr=0; jr<nc; jr+=NR ) {
              nr = min(nc-jr, NR);
              Cptr = &Ccol(ic+ir,jc+jr);

  	      if (mr == MR && nr == NR)
                ukr(kc, &alpha, &Ac[ir*kc], &Bc[jr*kc], &betaI, Cptr, ldC * sizeof(float));
              else {
	        Ctmp_th = &Ctmp[th_id * MR * NR];
	        ukr_edge(mr, nr, MR, NR, kc, &alpha, &Ac[ir*kc], &Bc[jr*kc], &betaI, Ctmp_th, Cptr, ldC);
	      }

            }
          }
        #ifdef OMP_ENABLE
        #pragma omp barrier
        #endif
	}
      }
    }

    } else if(loop == 5) {
    
    //=========================================== LOOP 5 PARALLELIZATION =======================================//
    for ( ic=0; ic<m; ic+=MC ) {
      mc = min(m-ic, MC);
      int its_mc = (int) ceil((double)mc/MR/omp_get_num_threads());

      for ( pc=0; pc<k; pc+=KC ) {
        kc = min(k-pc, KC);
        Aptr = &Acol(ic+its_mc*MR*th_id, pc);
      
        pack_RB( orderA, transA, min(its_mc*MR, mc - its_mc*MR*th_id), 
		 kc, Aptr, ldA, Ac + kc*its_mc*MR*th_id, MR);

        if ( pc==0 ) betaI = beta;
        else betaI = one;

	for ( jc=0; jc<n; jc+=NC ) {
          nc = min(n-jc, NC);
          int its_nc = (int) ceil((double)nc/NR/TH);

          Bptr = &Bcol(pc, jc + its_nc * NR * th_id);
          pack_CB( orderB, transB, kc, min(its_nc * NR, nc - its_nc * NR * th_id), 
                   Bptr, ldB, Bc + kc * its_nc * NR * th_id, NR);

        #ifdef OMP_ENABLE
        #pragma omp barrier
        #endif
          for ( ir=0; ir<mc; ir+=MR ) {
            mr = min(mc-ir, MR);

            for ( jr=th_id*(its_nc*NR); jr<min(nc,(th_id+1)*(its_nc*NR)); jr+=NR ) {
              nr = min(nc-jr, NR);
              Cptr = &Ccol(ic+ir,jc+jr);

  	      if (mr == MR && nr == NR)
                ukr(kc, &alpha, &Ac[ir*kc], &Bc[jr*kc], &betaI, Cptr, ldC * sizeof(float));
              else {
	        Ctmp_th = &Ctmp[th_id * MR * NR];
	        ukr_edge(mr, nr, MR, NR, kc, &alpha, &Ac[ir*kc], &Bc[jr*kc], &betaI, Ctmp_th, Cptr, ldC);
	      }

            }
          }
        #ifdef OMP_ENABLE
        #pragma omp barrier
        #endif
	}
      }
    }
    
    }

    #ifdef OMP_ENABLE
    }
    #endif
  }

}


void pack_RB( char orderM, char transM, int mc, int nc, DTYPE *M, int ldM, DTYPE *Mc, int RR ){
/*
  BLIS pack for M-->Mc
*/
  int    i, j, ii, k, rr;
  for ( i=0; i<mc; i+=RR ) { 
    k = i*nc;
    rr = min( mc-i, RR );
      for ( j=0; j<nc; j++ ) {
        for ( ii=0; ii<rr; ii++ ) {
	  Mc[k] = Mcol(i+ii, j);
          k++;
        } 
        k += (RR-rr);
      }
    }
}

void pack_CB( char orderM, char transM, int mc, int nc, DTYPE *M, int ldM, DTYPE *Mc, int RR ){
/*
  BLIS pack for M-->Mc
*/
  int    i, j, jj, k, nr;
  for ( j=0; j<nc; j+=RR ) { 
    k = j*mc;
    nr = min( nc-j, RR );
    for ( i=0; i<mc; i++ ) {
      for ( jj=0; jj<nr; jj++ ) {
        Mc[k] = Mcol(i,j+jj);
        k++;
      }
      k += (RR-nr);
    }
  }

}


void gemm_base_Cresident( char orderC, int m, int n, int k, 
                          DTYPE alpha, DTYPE *A, int ldA, 
                                       DTYPE *B, int ldB, 
                          DTYPE beta,  DTYPE *C, int ldC ){
/*
  Baseline micro-kernel 
  Replace with specialized micro-kernel where C-->m x n is resident in registers
*/
  int    i, j, p;
  DTYPE  zero = 0.0, tmp;

  for ( j=0; j<n; j++ )
    for ( i=0; i<m; i++ ) {
      tmp = 0.0; 
      for ( p=0; p<k; p++ ) 
        tmp += Acol(i,p) * Brow(p,j);

      if ( beta==zero ) {
        if ( orderC=='C' )
          Ccol(i,j) = alpha*tmp;
        else
          Crow(i,j) = alpha*tmp;
      }
      else {
        if ( orderC=='C' )
          Ccol(i,j) = alpha*tmp + beta*Ccol(i,j);
        else
          Crow(i,j) = alpha*tmp + beta*Crow(i,j);
      }
    }
}


