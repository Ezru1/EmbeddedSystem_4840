/* * Device driver for the VGA video generator
 *
 * A Platform device implemented using the misc subsystem
 *
 * Stephen A. Edwards
 * Columbia University
 *
 * References:
 * Linux source: Documentation/driver-model/platform.txt
 *               drivers/misc/arm-charlcd.c
 * http://www.linuxforu.com/tag/linux-device-drivers/
 * http://free-electrons.com/docs/
 *
 * "make" to build
 * insmod vga_ball.ko
 *
 * Check code style with
 * checkpatch.pl --file --no-tree vga_ball.c
 */
#include <linux/module.h>
#include <linux/init.h>
#include <linux/errno.h>
#include <linux/version.h>
#include <linux/kernel.h>
#include <linux/platform_device.h>
#include <linux/miscdevice.h>
#include <linux/slab.h>
#include <linux/io.h>
#include <linux/of.h>
#include <linux/of_address.h>
#include <linux/fs.h>
#include <linux/uaccess.h>
#include "vga_pixel.h"

#define DRIVER_NAME "vga_pixel"

/* Device registers */
#define BG_LUM(x) ((x)+0)
//register for x and y
#define AXIS(x) ((x)+4)


/*
 * Information about our device
 */
struct vga_pixel_dev {
	struct resource res; /* Resource: our registers */
	void __iomem *virtbase; /* Where registers can be accessed in memory */
        vga_pixel_color_t background;
        vga_pixel_axis_t position;
} dev;

/*
 * Write segments of a single digit
 * Assumes digit is in range and the device information has been set up
 */
static void write_background(vga_pixel_color_t *background)
{
	iowrite32(background->lum, BG_LUM(dev.virtbase) );
	dev.background = *background;
}

static void write_pixel(vga_pixel_axis_t *position)
{
	
	iowrite32(position->axis, AXIS(dev.virtbase) );
        dev.position = *position;
}

/*
 * Handle ioctl() calls from userspace:
 * Read or write the segments on single digits.
 * Note extensive error checking of arguments
 */
static long vga_pixel_ioctl(struct file *f, unsigned int cmd, unsigned long arg)
{
	vga_pixel_arg_t vla;

	switch (cmd) {
	case VGA_PIXEL_WRITE_BACKGROUND:
		if (copy_from_user(&vla, (vga_pixel_arg_t *) arg,
				   sizeof(vga_pixel_arg_t)))
			return -EACCES;
		write_background(&vla.background);
		break;

	case VGA_PIXEL_READ_BACKGROUND:
	  	vla.background = dev.background;
		if (copy_to_user((vga_pixel_arg_t *) arg, &vla,
				 sizeof(vga_pixel_arg_t)))
			return -EACCES;
		break;
	case VGA_PIXEL_WRITE_POSITION:
		if (copy_from_user(&vla, (vga_pixel_arg_t *) arg,
				   sizeof(vga_pixel_arg_t)))
			return -EACCES;
		write_pixel(&vla.position);
		break;

	case VGA_PIXEL_READ_POSITION:
	  	vla.position = dev.position;
		if (copy_to_user((vga_pixel_arg_t *) arg, &vla,
				 sizeof(vga_pixel_arg_t)))
			return -EACCES;
		break;

	default:
		return -EINVAL;
	}

	return 0;
}

/* The operations our device knows how to do */
static const struct file_operations vga_pixel_fops = {
	.owner		= THIS_MODULE,
	.unlocked_ioctl = vga_pixel_ioctl,
};

/* Information about our device for the "misc" framework -- like a char dev */
static struct miscdevice vga_pixel_misc_device = {
	.minor		= MISC_DYNAMIC_MINOR,
	.name		= DRIVER_NAME,
	.fops		= &vga_pixel_fops,
};

/*
 * Initialization code: get resources (registers) and display
 * a welcome message
 */
static int __init vga_pixel_probe(struct platform_device *pdev)
{
        vga_pixel_color_t beige = { 0x00};
        vga_pixel_axis_t start_pos = {0,0};
	int ret;

	/* Register ourselves as a misc device: creates /dev/vga_ball */
	ret = misc_register(&vga_pixel_misc_device);

	/* Get the address of our registers from the device tree */
	ret = of_address_to_resource(pdev->dev.of_node, 0, &dev.res);
	if (ret) {
		ret = -ENOENT;
		goto out_deregister;
	}

	/* Make sure we can use these registers */
	if (request_mem_region(dev.res.start, resource_size(&dev.res),
			       DRIVER_NAME) == NULL) {
		ret = -EBUSY;
		goto out_deregister;
	}

	/* Arrange access to our registers */
	dev.virtbase = of_iomap(pdev->dev.of_node, 0);
	if (dev.virtbase == NULL) {
		ret = -ENOMEM;
		goto out_release_mem_region;
	}
        
	/* Set an initial color */
        write_background(&beige);

        write_pixel(&start_pos);

	return 0;

out_release_mem_region:
	release_mem_region(dev.res.start, resource_size(&dev.res));
out_deregister:
	misc_deregister(&vga_pixel_misc_device);
	return ret;
}

/* Clean-up code: release resources */
static int vga_pixel_remove(struct platform_device *pdev)
{
	iounmap(dev.virtbase);
	release_mem_region(dev.res.start, resource_size(&dev.res));
	misc_deregister(&vga_pixel_misc_device);
	return 0;
}

/* Which "compatible" string(s) to search for in the Device Tree */
#ifdef CONFIG_OF
static const struct of_device_id vga_pixel_of_match[] = {
	{ .compatible = "csee4840,vga_pixel-1.0" },
	{},
};
MODULE_DEVICE_TABLE(of, vga_pixel_of_match);
#endif

/* Information for registering ourselves as a "platform" driver */
static struct platform_driver vga_pixel_driver = {
	.driver	= {
		.name	= DRIVER_NAME,
		.owner	= THIS_MODULE,
		.of_match_table = of_match_ptr(vga_pixel_of_match),
	},
	.remove	= __exit_p(vga_pixel_remove),
};

/* Called when the module is loaded: set things up */
static int __init vga_pixel_init(void)
{
	pr_info(DRIVER_NAME ": init\n");
	return platform_driver_probe(&vga_pixel_driver, vga_pixel_probe);
}

/* Calball when the module is unloaded: release resources */
static void __exit vga_pixel_exit(void)
{
	platform_driver_unregister(&vga_pixel_driver);
	pr_info(DRIVER_NAME ": exit\n");
}

module_init(vga_pixel_init);
module_exit(vga_pixel_exit);

MODULE_LICENSE("GPL");
MODULE_AUTHOR("Stephen A. Edwards, Columbia University");
MODULE_DESCRIPTION("VGA pixel driver");
