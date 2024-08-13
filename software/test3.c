/*
 * Userspace program that communicates with the aud and vga_zylo device driver
 * through ioctls
 * radomly generates notes at top of screen at fixed intervals
 * reads from hardware the detected note and checks if it matches the note currently in the green zone
 * Alex Yu, Rajat Tyagi, Sienna Brent, Riona Westphal
 * Columbia University
 */

#include <stdio.h>
#include <math.h>
#include "interfaces.h"
#include "vga_pixel.h"
#include "aud.h"
#include <sys/ioctl.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <string.h>
#include <unistd.h>
#include <stdlib.h>
#include <time.h>

#define X_MAX 639 
#define Y_MAX 479
#define WIDTH 640
#define HEIGHT 480
#define FRAMELENGTH     10
#define FRAMEVACLENGTH  0
#define FRAMETOTLENGTH  (FRAMELENGTH + FRAMEVACLENGTH)
#define FRAMENUMBERS    40
#define ANGLE           45
#define PLOTHEIGHT      360
#define PLOTWIDTH       600
#define DOWNSCALE       3
#define DOWNSCALE_AMP   130
#define MXH		150
#define XBIAS           30
#define DELAY		300


int vga_pixel_fd;
int vga_zylo_fd;
int aud_fd;


void set_pixel_axis(vga_pixel_axis_t *p){
  vga_pixel_arg_t vla;
  vla.position = *p;
  if (ioctl(vga_pixel_fd, VGA_PIXEL_WRITE_POSITION, &vla)) {
      perror("ioctl(VGA_PIXEL_SET_POSITION) failed");
      return;
  }
}

void set_background_color(vga_pixel_color_t *c)
{
  vga_pixel_arg_t vla;
  vla.background = *c;
  if (ioctl(vga_pixel_fd, VGA_PIXEL_WRITE_BACKGROUND, &vla)) {
      perror("ioctl(VGA_PIXEL_SET_BACKGROUND) failed");
      return;
  }
}


void f(int** CUR, int** Frames){
    double rad = ANGLE * (M_PI / 180.0);
    for(int i = 0; i < FRAMENUMBERS; i++){
        for(int j = 511;j > -1; j--){
            if (Frames[i][j] == 0) continue;
            int x = (int) (FRAMETOTLENGTH*i + (j/DOWNSCALE) * cos(rad));
            int y = (int) (Frames[i][j] + (j/DOWNSCALE) * sin(rad));
            if(x < 0 || y < 0) continue;
            for(int k = y - Frames[i][j]; k <= y; k++) 
                //for(int m = 0; m < FRAMELENGTH; m++)
                    CUR[x][k] = 255 - MXH + k - y + Frames[i][j];
        }
    }
    return;
}

void init_CUR(int** CUR){
    for(int i = 0; i < PLOTWIDTH; i++)
        for(int j = 0; j < PLOTHEIGHT; j++)
            CUR[i][j] = 0;
    return;
}

void clear_Sc(vga_pixel_axis_t position, vga_pixel_color_t tmp){
    for(int i = 0; i < 640; i++){
    	for(int j = 0; j < 480; j++){
       	    position.axis = (i << 16) + j;
	    tmp.lum = 0;
	    set_background_color(&tmp);
            set_pixel_axis(&position);

	}
    }
}

int max(int a, int b) {
    return (a > b) ? a : b;
}
int min(int a, int b) {
    return (a > b) ? b : a;
}


// simple game of hitting random falling notes when they reach the green zone
int main()
{
    

	aud_arg_t aat;
	aud_mem_t amt;
	vga_pixel_axis_t position;
	vga_pixel_arg_t vla;
	vga_pixel_color_t tmp;
	srand(time(NULL));

	static const char filename1[] = "/dev/vga_pixel";	
	static const char filename2[] = "/dev/aud";

	printf("VGA zylo test Userspace program started\n");
	printf("%d\n", sizeof(int));	
	printf("%d\n", sizeof(short));

	printf("VGA PIXEL Userspace program started\n");

	if ( (vga_pixel_fd = open(filename1, O_RDWR)) == -1) {
		fprintf(stderr, "could not open %s\n", filename1);
		return -1;
	}
	if ((aud_fd = open(filename2, O_RDWR)) == -1) {
		fprintf(stderr, "could not open %s\n", filename2);
		return -1;
	}
	
	
	
	int counter = 0; 	
	int MAX_NOTE_COUNT = 500 * 512;
	printf("start.");
    	int** Frames;
    	int** B;
    	Frames = (int**)malloc(FRAMENUMBERS * sizeof(int *));
    	for(int i = 0; i < FRAMENUMBERS; i++)
           Frames[i] = (int*)malloc(512 * sizeof(int));
    	B = (int**)malloc(PLOTWIDTH * sizeof(int *));
    	for(int i = 0; i < PLOTWIDTH; i++)
           B[i] = (int*)malloc(PLOTHEIGHT * sizeof(int));
	for(int i = 0; i < FRAMENUMBERS; i++)
           for(int j = 0; j < 512; j++)
        	Frames[i][j] = 0;
	int framecount = 0;
	int* N;
	while (counter < MAX_NOTE_COUNT + 5) {
        	amt.data = get_aud_data(aud_fd);
		//pause to let hardware catch up
		int x;
		printf("%d\n",amt.data);
		x = amt.data/DOWNSCALE_AMP;
		if (x > MXH) x = MXH;
		if (x < 0) {x = 0;printf("1\n");}
		if (counter % 512 == 0)
		   N = (int*)malloc(512 * sizeof(int));
		N[counter % 512] = x;
		if (counter % 512 == 511){
		   for (int i = 0; i < FRAMENUMBERS-1; i++)
			Frames[i] = Frames[i+1];
		   Frames[FRAMENUMBERS-1] = N;
		   init_CUR(B);
		   f(B,Frames);
		   //clear_Sc(position, tmp);
		   for (int j = PLOTWIDTH-4; j > -1; j -= 4) {
		   	for (int i = PLOTHEIGHT-1 ; i > -1 ; i--) {
                   	//position.axis = PLOTHEIGHT-1-i;//y
	                        position.axis = (j+XBIAS << 16) + PLOTHEIGHT-1-i; //x

				tmp.lum = 0;
				for(int k = 0; k < 4; k++)
                                    tmp.lum += B[j+k][i] << 8*k;
				set_background_color(&tmp);
                                set_pixel_axis(&position);
                        }
                   }
		}		
		counter++;
		usleep(DELAY);
	}
	return 0;
}
