#!/bin/bash

source ukr-generator.sh $1

git submodule update --init

#--------------------------------------------------------------
#BLIS INSTALL
#--------------------------------------------------------------
blis_install=$(cat Makefile.inc | grep BLIS_ENABLE | grep T)

if [ ! -z "$blis_install" ] ; then
  if [ "$ARCH" == "$RISCV" ] ; then 
    echo "WARNING: BLIS enabled, but not supported for RISCV"
  else
    blis_lib="src/blis/install/lib/libblis.a"
    if [[ ! -f $blis_lib ]]; then
      cd src/blis && mkdir -p install
      blis_path=$(pwd)"/install"
      ./configure  --prefix=${blis_path} -t openmp auto
      make -j 4 && make install
      cd ../..
    fi
  fi
fi
#--------------------------------------------------------------
#--------------------------------------------------------------


#--------------------------------------------------------------
#OPENBLAS INSTALL
#--------------------------------------------------------------
openblas_install=$(cat Makefile.inc | grep OPENBLAS_ENABLE | grep T)

if [ ! -z "$openblas_install" ] ; then
  openblas_lib="src/OpenBLAS/install/lib/libopenblas.a"
  if [[ ! -f $openblas_lib ]]; then
    cd src/OpenBLAS && mkdir -p install
    openblas_path=$(pwd)"/install"
    if [ "$ARCH" == "$RISCV" ] ; then 
      make USE_OPENMP=1 HOSTCC=gcc TARGET=C910V CC=riscv64-unknown-linux-gnu-gcc FC=riscv64-unknown-linux-gnu-gfortran \
	      && make install USE_OPENMP=1 HOSTCC=gcc TARGET=C910V CC=riscv64-unknown-linux-gnu-gcc FC=riscv64-unknown-linux-gnu-gfortran PREFIX=${openblas_path}
    else
      make USE_OPENMP=1 && make install USE_OPENMP=1 PREFIX=${openblas_path}
    fi
    cd ../..
  fi
fi
#--------------------------------------------------------------
#--------------------------------------------------------------

make clean; make arch=$ARCH

