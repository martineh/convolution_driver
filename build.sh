#!/bin/bash


git submodule update --init

./ukr-generator.sh

#--------------------------------------------------------------
#BLIS INSTALL
#--------------------------------------------------------------

blis_install=$(cat Makefile.inc | grep BLIS_ENABLE | grep T)

if [ ! -z "$blis_install" ] ; then
  blis_lib="src/blis/install/lib/libblis.a"
  if [[ ! -f $blis_lib ]]; then
    cd src/blis && mkdir -p install
    blis_path=$(pwd)"/install"
    ./configure  --prefix=${blis_path} -t openmp auto
    make -j 4 && make install
    cd ../..
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
    make USE_OPENMP=1 && make install PREFIX=${openblas_path}
    cd ../..
  fi
fi
#--------------------------------------------------------------
#--------------------------------------------------------------

make clean; make

