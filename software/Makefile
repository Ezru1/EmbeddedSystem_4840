ifneq (${KERNELRELEASE},)

# KERNELRELEASE defined: we are being compiled as part of the Kernel
        obj-m := vga_pixel.o aud.o

else

# We are being compiled as a module: use the Kernel build system

	KERNEL_SOURCE := /usr/src/linux-headers-$(shell uname -r)
        PWD := $(shell pwd)

default: module test3

c: c.c
	gcc -o c c.c

hello: hello.c
	gcc -o hello hello.c
test: test.c
	gcc -o test test.c interfaces.c interfaces.h
test3: test3.c
	gcc -o test3 test3.c interfaces.c interfaces.h -lm
module:
	${MAKE} -C ${KERNEL_SOURCE} SUBDIRS=${PWD} modules

clean:
	${MAKE} -C ${KERNEL_SOURCE} SUBDIRS=${PWD} clean
	${RM} hello test test2

TARFILES = Makefile README vga_pixel.h vga_pixel.c aud.c aud.h interfaces.c interfaces.h test.c hello.c c.c
TARFILE = lab3-sw.tar.gz
.PHONY : tar
tar : $(TARFILE)

$(TARFILE) : $(TARFILES)
	tar zcfC $(TARFILE) .. $(TARFILES:%=software/%)

endif 
