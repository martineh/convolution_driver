#!/bin/bash

TH_MAX=8

sed -i 's/.*BESTOF.*/BESTOF=T/' convolution.config

#-------------------------------------------------------------------------------
sed -i 's/.*ALGORITHM.*/ALGORITHM="CONVDIRECT"/' convolution.config
sed -i 's/.*THREADS.*/THREADS=1/' convolution.config

./convolution.sh cnn/Vgg16-imagenet output/convdirect_vgg16_1th.csv
./convolution.sh cnn/Resnet50-imagenet output/convdirect_resnet50_1th.csv

sed -i 's/.*THREADS.*/THREADS='$TH_MAX'/' convolution.config
./convolution.sh cnn/Vgg16-imagenet "output/convdirect_vgg16_"$TH_MAX"th.csv"
./convolution.sh cnn/Resnet50-imagenet "output/convdirect_resnet50_"$TH_MAX"th.csv"

#-------------------------------------------------------------------------------
sed -i 's/.*ALGORITHM.*/ALGORITHM="CONVGEMM"/' convolution.config
sed -i 's/.*THREADS.*/THREADS=1/' convolution.config

./convolution.sh cnn/Vgg16-imagenet output/convgemm_vgg16_1th.csv
./convolution.sh cnn/Resnet50-imagenet output/convgemm_resnet50_1th.csv

sed -i 's/.*THREADS.*/THREADS='$TH_MAX'/' convolution.config
./convolution.sh cnn/Vgg16-imagenet "output/convgemm_vgg16_"$TH_MAX"th.csv"
./convolution.sh cnn/Resnet50-imagenet "output/convgemm_resnet50_"$TH_MAX"th.csv"

#-------------------------------------------------------------------------------
sed -i 's/.*ALGORITHM.*/ALGORITHM="LOWERING"/' convolution.config
sed -i 's/.*GEMM.*/GEMM="B3A2C0"/' convolution.config
sed -i 's/.*THREADS.*/THREADS=1/' convolution.config

./convolution.sh cnn/Vgg16-imagenet output/lowering_B3A2C0_vgg16_1th.csv
./convolution.sh cnn/Resnet50-imagenet output/lowering_B3A2C0_resnet50_1th.csv

sed -i 's/.*THREADS.*/THREADS='$TH_MAX'/' convolution.config

./convolution.sh cnn/Vgg16-imagenet "output/lowering_B3A2C0_vgg16_"$TH_MAX"th.csv"
./convolution.sh cnn/Resnet50-imagenet "output/lowering_B3A2C0_resnet50_"$TH_MAX"th.csv"


#-------------------------------------------------------------------------------
sed -i 's/.*BESTOF.*/BESTOF=F/' convolution.config
sed -i 's/.*THREADS.*/THREADS=1/' convolution.config
sed -i 's/.*GEMM.*/GEMM="BLIS"/' convolution.config

./convolution.sh cnn/Vgg16-imagenet output/lowering_BLIS_vgg16_1th.csv
./convolution.sh cnn/Resnet50-imagenet output/lowering_BLIS_resnet50_1th.csv

sed -i 's/.*THREADS.*/THREADS='$TH_MAX'/' convolution.config

./convolution.sh cnn/Vgg16-imagenet "output/lowering_BLIS_vgg16_"$TH_MAX"th.csv"
./convolution.sh cnn/Resnet50-imagenet "output/lowering_BLIS_resnet50_"$TH_MAX"th.csv"

#-------------------------------------------------------------------------------

sed -i 's/.*BESTOF.*/BESTOF=F/' convolution.config
sed -i 's/.*THREADS.*/THREADS=1/' convolution.config
sed -i 's/.*GEMM.*/GEMM="OPENBLAS"/' convolution.config

./convolution.sh cnn/Vgg16-imagenet output/lowering_OPENBLAS_vgg16_1th.csv
./convolution.sh cnn/Resnet50-imagenet output/lowering_OPENBLAS_resnet50_1th.csv

sed -i 's/.*THREADS.*/THREADS='$TH_MAX'/' convolution.config

./convolution.sh cnn/Vgg16-imagenet "output/lowering_OPENBLAS_vgg16_"$TH_MAX"th.csv"
./convolution.sh cnn/Resnet50-imagenet "output/lowering_OPENBLAS_resnet50_"$TH_MAX"th.csv"

#-------------------------------------------------------------------------------


