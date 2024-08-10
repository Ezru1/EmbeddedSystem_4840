cmd_/root/Z/aud.ko := ld -r  -EL -T ./scripts/module-common.lds -T ./arch/arm/kernel/module.lds  --build-id  -o /root/Z/aud.ko /root/Z/aud.o /root/Z/aud.mod.o ;  true
