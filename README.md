# ConvLIB
The ConvLib performs convolution operations using the optimal algorithm configuration and hardware-aware code for the best performance.

## Requisites
- OpenMP if a parallel execution is required.

## Supported Hardware
- ARM A57, A78AE, CARMEL
- RISC-V Xuantie C906, C910

## How to install
1. Modify the `Makefile.inc` file for configuring the installation.
2. Configure the micro-kernel generation process in `SIMD_generator.config` file.
3. Run the `build.sh` script seleccting the architecture type as follows:
   ``` sh
   $ ./build.sh armv8
   ```

## How to use 
1. Configure the convolution features in the `convolution.config` file.
2. Run the `convolution.sh` script as follows:
   ``` sh
   $ ./convolution.sh cnn/MODEL output/OUT
   ```
Where `MODEL` is the desired CNN model and `OUT` is the name of the output file.

## Adding new CNN model
Adding a new CNN model is as easy as adding a new file to the `cnn` folder following the format of already existing ones. 

## Adding new hardware
1. Add a new file in the `cache-arch` folder following the `cache-TEMPLATE` file.
2. Add a new file in the `SIMD-arch` folder following the `SIMD-TEMPLATE` file.

## How to cite
Pending
