#!/bin/bash

source directConv.conf

#export OMP_BIND=true
#export OMP_PLACES="{0,1,2,3,4,5,6,7}"
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
   mkdir $OUTPATH
else
   RUNID=$(ls $OUTPATH | wc -l)
fi

if [ $# -eq 1 ]; then
  OUTCSV="$OUTPATH/run$RUNID-Logs.csv"
else
  OUTCSV="$2"
fi

#-Check block shalom and blocked tzemeng implementation for each arch.
if [ "$ARCH" = "RISCV" ]; then
  if [ "$ALGORITHM" = "BLOCKED_SHALOM" ] || [ "$ALGORITHM" = "BLOCKED_TZEMENG" ]; then
    echo "ERROR: '$ALGORITHM' not implemented for Risc-V"
    exit -1
  fi
else
  if [ "$ALGORITHM" = "BLOCKED_SHALOM" ] || [ "$ALGORITHM" = "BLOCKED_TZEMENG" ]; then
    echo "WARNING: Micro-kernel (7x12) only avaible for 'BLOCKED_SHALOM' and 'BLOCKED_TZEMENG'"
    MR=7
    NR=12
  fi
fi

#Correct MR and NR values
if [[ "$ALGORITHM" = "LOWERING" || "$ALGORITHM" = "CONVGEMM" ]]; then
  _MR=$MR
  _NR=$NR
else
  _MR=$NR
  _NR=$MR
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

#-Generate micro-kernel
if [[ "$ALGORITHM" = "LOWERING"  ||  "$ALGORITHM" = "BLOCKED_BLIS"  ||  "$ALGORITHM" = "CONVGEMM" ]] ; then
  if [ "$ARCH" = "RISCV" ]; then
    #Only for riscv simulator Possible options: cache="--ic=1024:4:32 --dc=512:4:32 --l2=2048:16:64"
    ./src/asm_generator/ukernel_generator_asm.py --mr ${_MR} --nr ${_NR} --arch riscv --op_b ${OP,,} --reorder ${_PIPELINING} ${_REORDER} ${_UNROLL}
  else
    ./src/asm_generator/ukernel_generator_asm.py --mr ${_MR} --nr ${_NR} --arch armv8 ${_PIPELINING} ${_REORDER} ${_UNROLL}
  fi
else
  if [ "$ARCH" = "RISCV" ]; then
    ./src/asm_generator/ukernel_generator_asm.py --mr 4 --nr 4 --arch riscv 
  else
    ./src/asm_generator/ukernel_generator_asm.py --mr 4 --nr 4 --arch armv8 
  fi
fi

echo

make clean
make MR=$MR NR=$NR SIMD=$ARCH ALG=$ALGORITHM GEMM=$GEMM TH=$THREADS
taskset -c $cpus ./build/test_convdirect.x "cnn" $CONFIGFILE $TMIN $TEST $DEBUG $OUTCSV NHWC

