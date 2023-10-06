#!/bin/bash

source directConv.conf

export OMP_PROC_BIND=TRUE
export OMP_NUM_THREADS=$THREADS

#BLIS
export BLIS_JC_NT=1
export BLIS_IC_NT=$THREADS
export BLIS_JR_NT=1
export BLIS_IR_NT=1

#OpenBLAS
export OPENBLAS_NUM_THREADS=$THREADS
export GOTO_NUM_THREADS=$THREADS

echo 
echo "Starting Driver for Direct Convolution..."
echo 

CONFIGFILE=$1
OUTPATH="output"

BUILDPATH="build"
mkdir -p $BUILDPATH

BIN="$BUILDPATH/convolution_driver.x"


RUNID=0
cpus="0"
if [ $THREADS -gt 0 ]; then
  if [ $THREADS -ne 1 ]; then
    cpus="0-$(($THREADS-1))"
  fi
else
    echo "ERROR: Minimum value for Number of thread is 1."
    exit -1
fi

if [ ! -f $CONFIGFILE ]; then
    echo "ERROR: The Test configure doesn't exist. Please, enter a valid filename."
    exit -1
fi

if [ ! -d $OUTPATH ]; then
   mkdir -p $OUTPATH
else
   RUNID=$(ls $OUTPATH | wc -l)
fi

if [ $# -eq 1 ]; then
  OUTCSV="$OUTPATH/run$RUNID-Logs.csv"
else
  OUTCSV="$2"
fi


#-Set pipelining option for ukernel generator
_PIPELINING=""
if [ "$PIPELINING" = "T" ]; then
  _PIPELINING="--pipelining"
  UNROLL=0
fi

#-Set reorder loads option for ukernel generator
_REORDER=""
if [ "$REORDER" = "T" ]; then
  _REORDER="--pipelining"
fi

#-Set reorder loads option for ukernel generator
_UNROLL=""
if [ $UNROLL -ne 0 ]; then
  _UNROLL="--unroll ${UNROLL}"
fi

if [ ! -f $BIN ]; then

  #-Generate micro-kernel
  if [ "$ARCH" = "RISCV" ]; then
    ./src/asm_generator/ukernels_generator.py --arch riscv \
  	                                      --op_b ${OP,,} --reorder ${_PIPELINING} ${_REORDER} ${_UNROLL}
  else
    ./src/asm_generator/ukernels_generator.py --arch armv8 \
  	                                      ${_PIPELINING} ${_REORDER} ${_UNROLL}
  fi
    
  echo
  
fi

make SIMD=$ARCH

taskset -c $cpus ./$BIN "cnn" $CONFIGFILE $TMIN $TEST $DEBUG $OUTCSV $MR $NR $THREADS $ALGORITHM $GEMM

