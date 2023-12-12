# ConvLIB
The ConvLib performs convolution operations using the optimal algorithm configuration and hardware-aware code for the best performance.

## Requisites
- OpenBLAS and BLIS (added as GitHub submodules).
- OpenMP if a parallel execution is required.

## Supported Hardware
- ARM A57, A78AE, CARMEL
- RISC-V Xuantie C906, C910

## How to install
1. Modify the `Makefile.inc` file for configuring the installation.
2. Configure the micro-kernel generation process in `SIMD_generator.config` file.
3. Run the `convolution.sh` script as follows:
   ``` sh
   ./build.sh
   ```

## How to use 
1. Configure the convolution features in the `convolution.config` file.
2. Run the `convolution.sh` script as follows:
   ``` sh
   ./convolution.sh
   ```
## Adding a new CNN model
Adding a new CNN model is as easy as adding a new file to the `cnn` folder following the format of already existent ones. 

## How to cite
Pending
