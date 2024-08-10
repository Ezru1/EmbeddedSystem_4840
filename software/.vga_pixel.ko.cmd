cmd_/root/Z/vga_pixel.ko := ld -r  -EL -T ./scripts/module-common.lds -T ./arch/arm/kernel/module.lds  --build-id  -o /root/Z/vga_pixel.ko /root/Z/vga_pixel.o /root/Z/vga_pixel.mod.o ;  true
