# Running tt-mlir pykernel on macOS

Quick guide for building and running pykernel on macOS without runtime support (no hardware execution).

**Setup**

You'll need a system descriptor file. Grab one from a real system or use a mock one:
```bash
export SYSTEM_DESC_PATH=$(pwd)/system_desc.ttsys
```

**Build Configuration**

Configure with runtime disabled:
```bash
cmake -G Ninja -B build \
    -DTTMLIR_ENABLE_RUNTIME=OFF \
    -DTTMLIR_ENABLE_STABLEHLO=OFF \
    -DTT_RUNTIME_ENABLE_PERF_TRACE=OFF \
    -DTTMLIR_ENABLE_BINDINGS_PYTHON=ON \
    -DTTMLIR_ENABLE_DEBUG_STRINGS=OFF \
    -DTTMLIR_ENABLE_EXPLORER=OFF \
    -DTTMLIR_ENABLE_RUNTIME_TESTS=OFF \
    -DTTMLIR_ENABLE_PYKERNEL=ON \
    -DCMAKE_BUILD_TYPE=Debug \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
    -DCMAKE_BUILD_PARALLEL_LEVEL=4 \
    -DFLATBUFFERS_COMPILER=/opt/homebrew/bin/flatc
```

**Build**

```bash
source env/activate
cmake --build build
```

**Run Example**

```bash
export SYSTEM_DESC_PATH=$(pwd)/system_desc.ttsys
source env/activate
python test/pykernel/gen/custom_dm_matmul.py
```

**What works**

The full compilation pipeline (Python → D2M → MLIR → TTKernel → EmitC) will run successfully. Hardware execution will not work due to no runtime support, but you can develop and test the compiler itself.

- Full MLIR compilation pipeline
- Kernel code generation
- All compiler passes
- No actual hardware execution (requires runtime + Tenstorrent hardware)
