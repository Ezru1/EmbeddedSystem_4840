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

/* Read and print the pixel position */
void print_pixel_axis() {
  vga_pixel_arg_t vla;
  if(ioctl(vga_pixel_fd, VGA_PIXEL_READ_POSITION, &vla)) {
      perror("ioctl(VGA_PIXEL_READ_POSITION) failed");
      return;
  }
  printf("position: %02d %02d %2d\n",vla.position.x1_axis, vla.position.x2_axis,vla.position.y2_axis);
}

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

/* Read and print the background color */
void print_background_color() {
  vga_pixel_arg_t vla;
  
  if (ioctl(vga_pixel_fd, VGA_PIXEL_READ_BACKGROUND, &vla)) {
      perror("ioctl(VGA_PIXEL_READ_BACKGROUND) failed");
      return;
  }
  //printf("color: %02x %02x %02x\n",vla.background.red, vla.background.green, vla.background.blue);
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
    FILE *coe_file = fopen("rs.coe", "r");
    if (!coe_file) {
        printf("Error opening image_pixel.coe\n");
        return NULL;
    }
    // Move file pointer to the end of the file
//    fseek(coe_file, 0, SEEK_END);

    // Get the current position of the file pointer, which is the length of the file
//    long file_length = ftell(coe_file);
//    printf("cc:%ld\n",file_length);
    // Assuming each pixel value is stored as a single integer in the COE file
    int** pixel_values = (int **)malloc(HEIGHT * sizeof(int *));
    
    // Read pixel values from the COE file
    for (int y = 0; y < HEIGHT; y++) {
      	pixel_values[y] = (int *)malloc(WIDTH * sizeof(int));
        //printf("%d\n",y);
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

  static const vga_pixel_color_t colors[] = {
    { 0xff, 0x00, 0x00 }, /* Red */
    { 0x00, 0xff, 0x00 }, /* Green */
    { 0x64, 0x97, 0xb1 }, /* Blue */
    { 0xff, 0xff, 0x00 }, /* Yellow */
    { 0x00, 0xff, 0xff }, /* Cyan */
    { 0xff, 0x00, 0xff }, /* Magenta */
    { 0x80, 0x80, 0x80 }, /* Gray */
    { 0x00, 0x00, 0x00 }, /* Black */
    { 0xff, 0xff, 0xff }  /* White */
  };

  vga_pixel_axis_t position;



# define COLORS 9

  printf("VGA PIXEL Userspace program started\n");

  if ( (vga_pixel_fd = open(filename, O_RDWR)) == -1) {
    fprintf(stderr, "could not open %s\n", filename);
    return -1;
  }

  printf("initial state: ");
  print_background_color();
  print_pixel_axis();
  pixel_values =  image_vga();
  
  position.x1_axis = 0;
  position.x2_axis = 0;
  position.y1_axis = 0;
  position.y2_axis = 0;  
  for (int i = 0 ; i < 480 ; i++) {
    for (int j = 0; j < 640; j++) {
        vga_pixel_color_t tmp;
        int xj;
        if (j == 0) xj = 127;
        else if (j == 127) xj = 0;
        else xj = j;
        tmp.red = pixel_values[i][j];
        tmp.green = pixel_values[i][j];
        tmp.blue = pixel_values[i][j];
        
        print_background_color();
        x = (position.x1_axis << 8) + position.x2_axis;
        x = x + 1;
        position.x1_axis = x >> 8;
        position.x2_axis = x % 256;
        set_background_color(&tmp);
        set_pixel_axis(&position);
        if (position.y2_axis == 100){
          usleep(200);
          printf("%d\n",x);
        }
    }
    y = (position.y1_axis << 8) + position.y2_axis;
    y = y + 1;
    position.y1_axis = y >> 8;
    position.y2_axis = y % 256;
    position.x1_axis = 0;
    position.x2_axis = 0;
  }
  printf("VGA PIXEL Userspace program terminating\n");
  return 0;
}
