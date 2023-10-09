
include Makefile.inc

#------------------------------------------
#| COMPILERS                              |
#------------------------------------------
arch=$(shell uname -p)

ifneq ($(arch), aarch64)
	CC       = riscv64-unknown-elf-gcc
	CLINKER  = riscv64-unknown-elf-gcc
	#FLAGS   +=  -O3 -fopenmp -march=rv64imafdcv0p7_zfh_xtheadc -mabi=lp64d -mtune=c910
else
	CC       = gcc
	CLINKER  = gcc
	OPTFLAGS = -march=armv8-a -O3 -fopenmp -DCHECK -DARMV8 -DFP32
endif
#------------------------------------------

OBJDIR = build
BIN    = convolution_driver.x

#------------------------------------------
LIBS = -lm
INCLUDE += -I$(BLIS_HOME)/include/blis/ 
INCLUDE += -I$(OPENBLAS_HOME)/include/
#------------------------------------------

SRC_ASM_FILES = $(wildcard ./src/asm_generator/ukernels/*.S)
OBJ_ASM_FILES = $(patsubst ./src/asm_generator/ukernels/%.S, $(OBJDIR)/%.o, $(SRC_ASM_FILES))
OBJ_FILES += $(OBJ_ASM_FILES)

SRC_CONV_FILES = $(wildcard ./src/*.c)
OBJ_CONV_FILES = $(patsubst ./src/%.c, $(OBJDIR)/%.o, $(SRC_CONV_FILES))

SRC_GEMM_FILES = $(wildcard ./src/gemm/*.c)
OBJ_GEMM_FILES = $(patsubst ./src/gemm/%.c, $(OBJDIR)/%.o, $(SRC_GEMM_FILES))

SRC_CONVGEMM_FILES = $(wildcard ./src/convGemm/*.c)
OBJ_CONVGEMM_FILES = $(patsubst ./src/convGemm/%.c, $(OBJDIR)/%.o, $(SRC_CONVGEMM_FILES))
 
OBJ_FILES  = $(OBJDIR)/model_level.o $(OBJDIR)/selector_ukernel.o $(OBJDIR)/gemm_ukernel.o $(OBJ_CONV_FILES) $(OBJ_ASM_FILES) $(OBJ_GEMM_FILES) $(OBJ_CONVGEMM_FILES)


all: $(OBJDIR)/$(BIN)

$(OBJDIR)/$(BIN): $(OBJ_FILES)
	$(CLINKER) $(OPTFLAGS) -o $@ $^ $(OPENBLAS_HOME)/lib/libopenblas.a $(BLIS_HOME)/lib/libblis.a $(LIBS)

$(OBJDIR)/%.o: ./src/%.c
	$(CC) $(CFLAGS) $(OPTFLAGS) -c -o $@ $< $(INCLUDE) $(LIBS)

$(OBJDIR)/%.o: ./src/gemm/%.c
	$(CC) $(CFLAGS) $(OPTFLAGS) -c -o $@ $< $(INCLUDE) $(LIBS)

$(OBJDIR)/%.o: ./src/convGemm/%.c
	$(CC) $(CFLAGS) $(OPTFLAGS) -c -o $@ $< $(INCLUDE) $(LIBS)

$(OBJDIR)/%.o: ./src/asm_generator/ukernels/%.S
	$(CC) $(CFLAGS) $(OPTFLAGS) -c -o $@ $< $(INCLUDE) $(LIBS)

$(OBJDIR)/model_level.o: ./src/modelLevel/model_level.c 
	$(CC) $(CFLAGS) $(OPTFLAGS) -c -o $@ $< $(INCLUDE) $(LIBS)

$(OBJDIR)/gemm_ukernel.o: ./src/asm_generator/ukernels/gemm_ukernel.c
	$(CC) $(CFLAGS) $(OPTFLAGS) -c -o $@ $< $(INCLUDE) $(LIBS)

$(OBJDIR)/selector_ukernel.o: ./src/asm_generator/ukernels/selector_ukernel.c
	$(CC) $(CFLAGS) $(OPTFLAGS) -c -o $@ $< $(INCLUDE) $(LIBS)

clean:
	rm $(OBJDIR)/* 

