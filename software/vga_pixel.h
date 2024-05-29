#ifndef _VGA_PIXEL_H
#define _VGA_PIXEL_H

#include <linux/ioctl.h>

typedef struct {
	unsigned char red, green, blue;
} vga_pixel_color_t;
  
typedef struct {
        unsigned int x1_axis, x2_axis, y1_axis, y2_axis;
} vga_pixel_axis_t;

typedef struct {
  vga_pixel_color_t background;
  vga_pixel_axis_t position;
} vga_pixel_arg_t;

#define VGA_PIXEL_MAGIC 'q'

/* ioctls and their arguments */
#define VGA_PIXEL_WRITE_BACKGROUND _IOW(VGA_PIXEL_MAGIC, 1, vga_pixel_arg_t *)
#define VGA_PIXEL_READ_BACKGROUND  _IOR(VGA_PIXEL_MAGIC, 2, vga_pixel_arg_t *)
#define VGA_PIXEL_WRITE_POSITION _IOW(VGA_PIXEL_MAGIC, 3, vga_pixel_arg_t *)
#define VGA_PIXEL_READ_POSITION _IOR(VGA_PIXEL_MAGIC, 4, vga_pixel_arg_t *)
#endif
