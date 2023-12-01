#!/bin/bash

source SIMD_generator.config

CONFIG_PATH=SIMD-arch

sys_arch=$(uname -p)

if [ "$sys_arch" = "aarch64" ]; then
  ARCH="ARMV8"
else
  ARCH="RISCV"
fi

BUILDPATH="build"
mkdir -p $BUILDPATH

#-Set pipelining option for ukernel generator
_PIPELINING=""
if [ "$PIPELINING" = "T" ]; then
  _PIPELINING="--pipelining"
  UNROLL=0
fi

#-Set reorder loads option for ukernel generator
_REORDER=""
if [ "$REORDER" = "T" ]; then
  _REORDER="--reorder"
fi

#-Set reorder loads option for ukernel generator
_UNROLL=""
if [ $UNROLL -ne 0 ]; then
  _UNROLL="--unroll ${UNROLL}"
fi

#-Set reorder loads option for ukernel generator
_OP=""
if [ "$OP" != "NORMAL" ]; then
  _OP="--op_b ${OP,,}"
fi

if [ "$DATA" != "FP32" ]; then
  echo "ERROR: Actually, only is supported Float 32 bits data type."
  exit -1
fi


if [ ! -f $CONFIG_PATH/$ARCH_CONFIG ]; then
    echo "ERROR: The Architecture configure doesn't exist. Please, enter a valid filename."
    exit -1
fi

VLEN=$(tail -1 $CONFIG_PATH/$ARCH_CONFIG | cut -f1)
MAXVECTOR=$(tail -1 $CONFIG_PATH/$ARCH_CONFIG | cut -f2)

rm src/asm_generator/ukernels/*

if [ "$ARCH" = "RISCV" ]; then
  ./src/asm_generator/ukernels_generator.py --arch riscv --data ${DATA} --vlen ${VLEN} \
	                                    --maxvec ${MAXVECTOR} ${_OP} ${_REORDER} ${_PIPELINING} ${_UNROLL}
else
  ./src/asm_generator/ukernels_generator.py --arch armv8 --data ${DATA} --vlen ${VLEN} \
	                                    --maxvec ${MAXVECTOR} ${_PIPELINING} ${_UNROLL}
fi

