# Intel NPU Compiler Override

This directory contains `libnpu_driver_compiler.so`, tracked with Git LFS.

It is vendored so Pishi can bootstrap on another Lunar Lake laptop without manually extracting Intel's upstream NPU userspace package first.

The binary was copied from Intel's upstream Linux NPU Driver userspace package:

```text
intel-driver-compiler-npu_1.32.1.20260422-24767473183~ubuntu24.04_amd64.deb
```

SHA-256:

```text
565baeb6f1a1311e3d5160eef8a88ef6d0acfbd9aa844e171fa2b9f52ac1af41  libnpu_driver_compiler.so
```

The launcher prefers this vendored copy, then falls back to:

```text
~/.local/share/dictophone/npu-compiler/libnpu_driver_compiler.so
```

You can override either path with:

```bash
DICTOPHONE_NPU_COMPILER_DIR=/path/to/compiler-dir
```

