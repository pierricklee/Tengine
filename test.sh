#!/bin/bash 

QEMU=/opt/toolchains/csky_qemu/bin/qemu-riscv64
RISC_GDB=/opt/toolchains/riscv_linux/bin/riscv64-unknown-linux-gnu-gdb

# run
# $QEMU -g 2020 -L /home/pierrick/code/riscv/toolchains/riscv_linux/sysroot ./cjpeg -rgb -quality 95 -dct float -sample 2x1 -outfile ./out_img_pureC.jpg ../testimages/vgl_5674_0098.bmp
$QEMU -g 2233 -L /opt/toolchains/riscv_linux/sysroot ./build/tests/test_op_conv
# $QEMU -g 2020 -L /home/pierrick/code/riscv/toolchains/riscv_linux/sysroot ./cjpeg -rgb -quality 95 -dct float -sample 2x2 -outfile ./out_img_downsample_h2v2_rv64.jpg ../testimages/vgl_5674_0098.bmp

