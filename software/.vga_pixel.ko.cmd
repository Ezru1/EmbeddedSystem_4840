cmd_/root/aaa/vga_pixel.ko := ld -r  -EL -T ./scripts/module-common.lds -T ./arch/arm/kernel/module.lds  --build-id  -o /root/aaa/vga_pixel.ko /root/aaa/vga_pixel.o /root/aaa/vga_pixel.mod.o ;  true
