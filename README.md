# ConvLIB
The ConvLib performs convolution operations using the optimal algorithm configuration and hardware-aware code for the best performance.

## Requisites
- ConvLIB can interact with Linear Algebra Libraries such as OpenBLAS or BLIS so they are added as a submodule.
- As ConvLIB is a parallel library, an OpenMP solution is also required.

## Supported Hardware
- ARM A57, A78AE, CARMEL
- RISC-V Xuantie C906, C910

## How to install
1. Modify the `Makefile.inc` file for configuring the installation.
2. Configure the micro-kernel generation process in `SIMD_generator.config` file.
3. Run the `build.sh` script.

## How to use 
1. Configure the convolution features in the `convolution.config` file.
2. Run the `convolution.sh` script.
   
## How to cite
Pending
