
AUTO="auto"
ARMV8="armv8"
RISCV="riscv"

if [ $# -ne 1 ] ; then
  echo "USAGE ERROR: Architecture SIMD not specify. Possible values [$AUTO | $ARMV8 | $RISCV]"
  exit -1
fi

ARCH=$1

if [ "$ARCH" == "$AUTO" ]; then
  auto=$(uname -p)
  if [ "$auto" == "aarch64" ] ; then
    ARCH=$ARMV8
  elif [ "$auto" == "riscv64" ] ; then
    ARCH=$RISCV
  else
    echo "ERROR: Automatic Architecture detection fail or unsuported. Please set the architecture manually: [$ARMV8 | $RISCV]"
    exit -1
  fi
elif [ "$ARCH" != "$ARMV8" ] && [ "$ARCH" != "$RISCV" ] ; then
  echo "USAGE ERROR: Architecture SIMD unsuported. Possible values [$AUTO | $ARMV8 | $RISCV]"
  exit -1
fi

