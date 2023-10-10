#!/bin/bash

source ukr-generator.config

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

rm src/asm_generator/ukernels/*

if [ "$ARCH" = "RISCV" ]; then
  ./src/asm_generator/ukernels_generator.py --arch riscv --op_b ${OP,,} ${_PIPELINING} ${_REORDER} ${_UNROLL}
else
  ./src/asm_generator/ukernels_generator.py --arch armv8 ${_PIPELINING} ${_UNROLL}
fi

