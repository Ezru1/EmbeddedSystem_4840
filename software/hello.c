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
 // printf("%d,%d,%d,%d\n",vla.position.x1_axis,vla.position.x2_axis,vla.position.y1_axis,vla.position.y2_axis);
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

int** image_vga() {
    FILE *coe_file = fopen("i.coe", "r");
    if (!coe_file) {
        printf("Error opening image_pixel.coe\n");
        return NULL;
    }
    int** pixel_values = (int **)malloc(HEIGHT * sizeof(int *));
    
    for (int y = 0; y < HEIGHT; y++) {
      	pixel_values[y] = (int *)malloc(WIDTH * sizeof(int));
	for (int x = 0; x < WIDTH; x++) {
            if (fscanf(coe_file, "%d", &pixel_values[y][x]) != 1) {
                printf("Error reading pixel value at position (%d, %d)\n", x, y);
                fclose(coe_file);
                return NULL;
            }
            //printf("(%d, %d, %hhu)\n", y, x, pixel_values[y][x]);
        }
    }
    long file_length = ftell(coe_file);
    //printf("cc:%ld\n",file_length);

    // Close the COE file
    fclose(coe_file);

    // Now you have the pixel values stored in the pixel_values array
    // You can process them as needed
    
    return pixel_values;
}

int main()
{
  vga_pixel_arg_t vla;
  int i;
  int** pixel_values;
  int x,y;
  static const char filename[] = "/dev/vga_pixel";
  
  vga_pixel_axis_t position;



# define COLORS 9

  printf("VGA PIXEL Userspace program started\n");

  if ( (vga_pixel_fd = open(filename, O_RDWR)) == -1) {
    fprintf(stderr, "could not open %s\n", filename);
    return -1;
  }

  pixel_values =  image_vga();
  
  for (int j = 1; j < 640; j++) {
     for (int i = 0 ; i < 403 ; i++) {
    	position.axis = (639-j << 16)  + i;
	vga_pixel_color_t tmp;
        tmp.lum = pixel_values[i][639-j];
	set_background_color(&tmp);
        set_pixel_axis(&position);
         //if (i > 80 && i < 120){
          // usleep(20000);
          // printf("%d\n",i);
         //}
	usleep(80);
    }
  }
  printf("VGA PIXEL Userspace program terminating\n");
  return 0;
}
