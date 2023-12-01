#!/bin/bash

source convolution.config

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


CONFIGFILE=$1
OUTPATH="output"

mkdir -p $OUTPATH

RUNID=0
cpus="0"

if [ "$TEST" = "T" ]; then
  TMIN=0.0
fi

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

if [ ! -f $OUTCSV ]; then
  touch $OUTCSV
fi

if [ "$ALGORITHM" = "LOWERING" ] && [ "$GEMM" = "OPENBLAS" ] ; then
  echo "WARNING: LOWERING + OPENBLAS Enable. BESTOF Option not available. Disabled automaticaly."
  BESTOF=F
fi

if [ "$ALGORITHM" = "LOWERING" ] && [ "$GEMM" = "BLIS" ]; then
  echo "WARNING: LOWERING + BLIS Enable. BESTOF Option not available. Disabled automaticaly."
  BESTOF=F
fi

if [ "$ALGORITHM" = "WINOGRAD" ] ; then
  echo "WARNING: WINOGRAD Enable. BESTOF Option not available. Disabled automaticaly."
  BESTOF=F
  MR=4
  NR=4
fi

sys_arch=$(uname -p)
if [ "$sys_arch" = "aarch64" ]; then
  taskset -c $cpus ./build/convolution_driver.x "cnn" $CONFIGFILE $TMIN $TEST $DEBUG $OUTCSV $MR $NR $THREADS $ALGORITHM $GEMM $BESTOF "cache-arch/"$PLATFORM
else
  qemu-riscv64 -cpu c906fdv ./build/convolution_driver.x "cnn" $CONFIGFILE $TMIN $TEST $DEBUG $OUTCSV $MR $NR $THREADS $ALGORITHM $GEMM $BESTOF "cache-arch/"$PLATFORM
fi


