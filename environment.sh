#!/bin/bash

./ukr-generator.sh

git submodule update --init

#--------------------------------------------------------------
#BLIS INSTALL
#--------------------------------------------------------------

blis_lib="src/blis/install/lib/libblis.a"
if [[ ! -f $blis_lib ]]; then
  cd src/blis && mkdir -p install
  blis_path=$(pwd)"/install"
  ./configure  --prefix=${blis_path} -t openmp auto
  make -j 4 && make install
  cd ../..
fi

#--------------------------------------------------------------
#--------------------------------------------------------------


#--------------------------------------------------------------
#OPENBLAS INSTALL
#--------------------------------------------------------------
openblas_lib="src/OpenBLAS/install/lib/libopenblas.a"
if [[ ! -f $openblas_lib ]]; then
  cd src/OpenBLAS && mkdir -p install
  openblas_path=$(pwd)"/install"
  make USE_OPENMP=1 && make install PREFIX=${openblas_path}
  cd ../..
fi
#--------------------------------------------------------------
#--------------------------------------------------------------


