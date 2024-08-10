/*
 * Userspace program that communicates with the vga_pixel device driver
 * through ioctls
 *
 * Stephen A. Edwards
 * Columbia University
 */

#include <stdio.h>
#include <stdlib.h>
#include "vga_pixel.h"
#include <sys/ioctl.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <string.h>
#include <unistd.h>


#define WIDTH 640
#define HEIGHT 480
int vga_pixel_fd;


/* Set the pixel axis */
void set_pixel_axis(vga_pixel_axis_t *p){
  vga_pixel_arg_t vla;
  vla.position = *p;
  if (ioctl(vga_pixel_fd, VGA_PIXEL_WRITE_POSITION, &vla)) {
      perror("ioctl(VGA_PIXEL_SET_POSITION) failed");
      return;
  }
}


/* Set the background color */
void set_background_color(vga_pixel_color_t *c)
{
  vga_pixel_arg_t vla;
  vla.background = *c;
  if (ioctl(vga_pixel_fd, VGA_PIXEL_WRITE_BACKGROUND, &vla)) {
      perror("ioctl(VGA_PIXEL_SET_BACKGROUND) failed");
      return;
  }
}


int main()
{
  vga_pixel_arg_t vla;
  int i;
  int** pixel_values;
  int x,y;
  static const char filename[] = "/dev/vga_pixel";

  vga_pixel_axis_t position;
  vga_pixel_color_t tmp;




  printf("VGA PIXEL Userspace program started\n");

  if ( (vga_pixel_fd = open(filename, O_RDWR)) == -1) {
    fprintf(stderr, "could not open %s\n", filename);
    return -1;
  }
  
  for (int i = 0 ; i < 400 ; i++) {
    for (int j = 0; j < 160; j++) {
        tmp.lum = 0;
	for(int k = 0;k < 4; k++)
           tmp.lum += ((128+ i * 128) % 256) << (8*k);
	position.axis = i + (j << 18);
        set_background_color(&tmp);
        set_pixel_axis(&position);
	usleep(200000);
    }
  }
  printf("VGA PIXEL Userspace program terminating\n");
  return 0;
}
