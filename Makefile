
OPENBLAS_HOME=/home/martineh/software/OpenBLAS/install/
BLIS_HOME=/home/martineh/software/blis/install/

#------------------------------------------
#| COMPILERS                              |
#------------------------------------------
FLAGS = -O3 

ifneq ($(TH), 1)
	FLAGS += -fopenmp
endif

ifeq ($(SIMD), RISCV)
	CC       =  riscv64-unknown-elf-gcc
	CLINKER  =  riscv64-unknown-elf-gcc
	#FLAGS   +=  -O3  -Wall  -march=rv64imafdcv0p7_zfh_xtheadc -mabi=lp64d -mtune=c910
else
	CC       =  gcc
	CLINKER  =  gcc
	FLAGS   +=  -march=armv8-a
endif
#------------------------------------------

OBJDIR = build
BIN    = test_convdirect.x

#------------------------------------------
#| COMPILER FLAGS                         |
#------------------------------------------
OPTFLAGS = $(FLAGS) -DCHECK -DMR=$(MR) -DNR=$(NR) -DKR=1 $(MODE) -D$(SIMD) -D$(ALG) -D$(GEMM) -DTH=$(TH) -DFP32
LIBS = -lm

ifeq ($(GEMM), BLIS)
	LIBS    += -lblis -L$(BLIS_HOME)/lib/ 
	INCLUDE += -I$(BLIS_HOME)/include/blis/ 
else ifeq ($(GEMM), OPENBLAS)
	LIBS    += -lopenblas -L$(OPENBLAS_HOME)/lib/
	INCLUDE += -I$(OPENBLAS_HOME)/include/
endif
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
 
SRC_KERNEL_FILES = $(wildcard ./src/ukernels/*.c)
OBJ_KERNEL_FILES = $(patsubst ./src/ukernels/%.c, $(OBJDIR)/%.o, $(SRC_KERNEL_FILES))

OBJ_FILES  = $(OBJDIR)/model_level.o $(OBJDIR)/gemm_ukernel.o $(OBJ_CONV_FILES) $(OBJ_ASM_FILES) $(OBJ_KERNEL_FILES) $(OBJ_GEMM_FILES) $(OBJ_CONVGEMM_FILES)


all: $(OBJDIR)/$(BIN)

$(OBJDIR)/$(BIN): $(OBJ_FILES)
	$(CLINKER) $(OPTFLAGS) -o $@ $^ $(LIBS)

$(OBJDIR)/%.o: ./src/%.c
	$(CC) $(CFLAGS) $(OPTFLAGS) -c -o $@ $< $(INCLUDE) $(LIBS)

$(OBJDIR)/%.o: ./src/ukernels/%.c
	$(CC) $(CFLAGS) $(OPTFLAGS) -c -o $@ $< $(INCLUDE) $(LIBS)

$(OBJDIR)/%.o: ./src/gemm/%.c
	$(CC) $(CFLAGS) $(OPTFLAGS) -c -o $@ $< $(INCLUDE) $(LIBS)

$(OBJDIR)/%.o: ./src/convGemm/%.c
	$(CC) $(CFLAGS) $(OPTFLAGS) -c -o $@ $< $(INCLUDE) $(LIBS)

$(OBJDIR)/%.o: ./src/asm_generator/ukernels/%.S
	$(CC) $(CFLAGS) $(OPTFLAGS) -c -o $@ $< $(INCLUDE) $(LIBS)

$(OBJDIR)/%.o: ./src/asm_generator/ukernels/gemm_ukernel.c
	$(CC) $(CFLAGS) $(OPTFLAGS) -c -o $@ $< $(INCLUDE) $(LIBS)

$(OBJDIR)/model_level.o: ./src/modelLevel/model_level.c 
	@mkdir -p $(OBJDIR)
	$(CC) $(CFLAGS) $(OPTFLAGS) -c -o $@ $< $(INCLUDE) $(LIBS)

clean:
	rm $(OBJDIR)/* 

