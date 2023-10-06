import os

u_folder = "ukernels"
c_name   = "gemm_ukernel.c"
h_name   = "gemm_ukernels_headers.h"

def generate_general_call(asm):
    MR = asm.orig_mr
    NR = asm.orig_nr

    #Micro-Kernel Output Files
    name_path = os.path.dirname(__file__) + "/" + u_folder 

    file_name = name_path + "/" + c_name
    foutS = open(file_name, "w")
   
    foutS.write("\n#include <stdio.h>\n")
    foutS.write("#include <stdlib.h>\n")
    if asm.arch == "armv8":
        foutS.write("#include <arm_neon.h>\n")
    foutS.write("\n#include \"" + h_name + "\"\n\n")

    foutS.write("void gemm_ukernel_asm(size_t mr, size_t nr, size_t _MR, size_t _NR, size_t kc, float *alpha, \n\
                                       float *a, float *b, float *beta, float *ctmp, \n\
                                       float *C, size_t ldC) {\n")
    foutS.write("  int i, j;\n")
    foutS.write("  float  beta_edge = 0.0;\n\n")
    foutS.write("  if (mr == _MR && nr == _NR) {\n")
    foutS.write("    gemm_ukernel_asm_%dx%d(kc, alpha, a, b, beta, C, ldC * sizeof(float));\n" % (MR, NR))
    foutS.write("  } else {\n")
    foutS.write("    gemm_ukernel_asm_%dx%d(kc, alpha, a, b, &beta_edge, ctmp, _MR * sizeof(float));\n" % (MR, NR))
    foutS.write("    for (j = 0; j < nr; j++)\n")
    foutS.write("      for (i = 0; i < mr; i++)\n")
    foutS.write("        C[j*ldC + i] = (*beta) * C[j*ldC + i] + ctmp[j * _MR + i];\n")
    foutS.write("  }\n")
    foutS.write("}\n")
    foutS.close()


#TODO: I don't know... is 'generate_packA' a member function from the class ARMV8 or RISCV??
def generate_packA(asm):
    MR = asm.orig_mr
    NR = asm.orig_nr

    #Micro-Kernel Output Files
    name_path = os.path.dirname(__file__) + "/" + u_folder

    file_name = name_path + "/" + c_name
    foutS = open(file_name, "a")
  
    if asm.arch == "riscv":
        print("Waring: Pack vectorization not implemented for RISCV\n")
    else:
        MR = asm.orig_mr
        MR_row = MR // asm.vl

        foutS.write("\n")
        foutS.write("void pack_A( int _MR, int mc, int kc, float *A, int ldA, float *Ac) {\n")
        foutS.write("  int i, j, ii, k, rr;\n");

        Aregs = "A0"
        for i in range(1, MR_row):
            Aregs += ", A%d" % (i)

        foutS.write("  float32x4_t %s;\n" %(Aregs))
        foutS.write("  for ( i=0; i<mc; i+=_MR ) {\n")
        foutS.write("    k = i * kc;\n")
        foutS.write("    rr = mc-i < _MR ? mc-i : _MR;\n") #(((a)<(b))?(a):(b))
        foutS.write("    if (rr == _MR) {\n")
        foutS.write("      for ( j=0; j<kc; j++ ) {\n")

        dsp=0
        for i in range(0, MR_row):
            foutS.write("        A%d = vld1q_f32(&A[j * ldA + (i + %d)]);\n" %(i, dsp))
            dsp += asm.vl

        dsp=0
        for i in range(0, MR_row):
            foutS.write("        vst1q_f32(&Ac[k], A%d); k += 4;\n" % (i))
            dsp += asm.vl

        foutS.write("      }\n")
        foutS.write("    } else {\n")
        foutS.write("      for ( j=0; j<kc; j++ ) {\n")
        foutS.write("        for ( ii=0; ii < rr; ii++ ) {\n")
        foutS.write("          Ac[k] = A[j * ldA + (i + ii)];\n")
        foutS.write("          k++;\n")
        foutS.write("        }\n")
        foutS.write("        k += (_MR - rr);\n")
        foutS.write("      }\n")
        foutS.write("    }\n")
        foutS.write("  }\n")
        foutS.write("}\n")

    foutS.close()


def new_micro_file(MR, NR):
    #Micro-Kernel Output Files
    name_path = os.path.dirname(__file__) + "/" + u_folder
   
    #Include .S
    Cname = "gemm_ukernel_asm_%dx%d.S" % (MR, NR)
    file_name = name_path + "/" + Cname
    foutS = open(file_name, "w")
    foutS.write("\n#include \"gemm_ukernel_asm_%dx%d.h\"\n" % (MR, NR))
    foutS.close()

    #ASM Implementation
    Cname = "gemm_ukernel_asm_%dx%d.h" % (MR, NR)
    file_name = name_path + "/" + Cname

    return open(file_name, "w")


def add_header(MR, NR):
    #Micro-Kernel Output Files
    name_path = os.path.dirname(__file__) + "/" + u_folder
    file_name = name_path + "/" + h_name
   
    add_header = True
    if os.path.exists(file_name):
        fdout = open(file_name, "r")

        #check if header exists
        header_check = "gemm_ukernel_asm_%dx%d" % (MR, NR)
        for line in fdout:
            if header_check in line:
                add_header = False
                break

        if add_header:
            fdout.close()
            fdout = open(file_name, "a")
    else:
        fdout = open(file_name, "w")
        fdout.write("void gemm_ukernel_asm(size_t, size_t, size_t, size_t, size_t, float *, float *, float *, float *, float *, float *, size_t);\n")

    if add_header:
        fdout.write("void gemm_ukernel_asm_%dx%d(size_t , float *, float *, float *, float *, float *, size_t );\n" % (MR, NR))

    fdout.close()


def clear_path():
    name_path = os.path.dirname(__file__) + "/" + u_folder + "/"

    if os.path.exists(name_path):
        for f in os.listdir(name_path):
            file_name = name_path + f
            os.remove(file_name)
    else:
        os.makedirs(name_path)

