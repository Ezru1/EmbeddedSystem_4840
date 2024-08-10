/*
 * Userspace program that communicates with the aud and vga_zylo device driver
 * through ioctls
 * radomly generates notes at top of screen at fixed intervals
 * reads from hardware the detected note and checks if it matches the note currently in the green zone
 * Alex Yu, Rajat Tyagi, Sienna Brent, Riona Westphal
 * Columbia University
 */

#include <stdio.h>
#include "interfaces.h"
#include "vga_zylo.h"
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

int vga_pixel_fd;
int vga_zylo_fd;
int aud_fd;


void set_pixel_axis(vga_pixel_axis_t *p){
  vga_pixel_arg_t vla;
  vla.position = *p;
 // printf("%d,%d,%d,%d\n",vla.position.x1_axis,vla.position.x2_axis,vla.position.y1_axis,vla.position.y2_axis);
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

void updateBall(sprite *obj) {
	obj->x += obj->dx;
	obj->y += obj->dy;
	if (obj->x < 0 || obj->x >= X_MAX) {
		obj->y = 380;
		obj->x = 700;
		obj->id = 0;
		obj->dy = 0;
		obj->dx = 0;
    }
	if (obj->y < 0 || obj->y > Y_MAX) {
		// obj->dy = -obj->dy;
		obj->id = 0;
		obj->dy = 0;
	}
	// if () {
	// 	obj->dy = -obj->dy;
	// 	obj->id = 0;
	// }
}

void scorecombosetup(sprite *sprites) {
	//index 0 acts strangly
	//'SCORE'
	sprites[1].id = 17; //S
	sprites[2].id = 12; //C
	sprites[3].id = 15; //O
	sprites[4].id = 16; //R
	sprites[5].id = 13; //E
	for (int i = 1; i < 6; i++) {
		sprites[i].x = 480+32*(i-1); 
		sprites[i].y = 40;
		sprites[i].dx = 0;  
		sprites[i].dy = 0; 
		sprites[i].hit = 1;
		sprites[i].index = i;
	}
	sprites[6].id = 10; //0
	sprites[7].id = 10; //0
	sprites[8].id = 10; //0
	for (int i = 6; i < 9; i++) {
		sprites[i].x = 480+32+32*(i-6); 
		sprites[i].y = 90;
		sprites[i].dx = 0;  
		sprites[i].dy = 0; 
		sprites[i].hit = 1;
		sprites[i].index = i;
	}

	//'COMBO'
	sprites[9].id =  12; //C
	sprites[10].id = 15; //O
	sprites[11].id = 14; //M
	sprites[12].id = 11; //B
	sprites[13].id = 15; //O
	for (int i = 9; i < 14; i++) {
		sprites[i].x = 480+32*(i-9); 
		sprites[i].y = 140;
		sprites[i].dx = 0;  
		sprites[i].dy = 0; 
		sprites[i].hit = 1;
		sprites[i].index = i;
	}
	sprites[14].id = 10; //0
	sprites[15].id = 10; //0
	sprites[16].id = 10; //0
	for (int i = 14; i < 17; i++) {
		sprites[i].x = 480+32+32*(i-14); 
		sprites[i].y = 190;
		sprites[i].dx = 0;  
		sprites[i].dy = 0; 
		sprites[i].hit = 1;
		sprites[i].index = i;
	}
	
	//'MAX'
	sprites[17].id = 14; //M
	sprites[18].id = 26; //A
	sprites[19].id = 27; //X
	for (int i = 17; i < 20; i++) {
		sprites[i].x = 480+32+32*(i-17); 
		sprites[i].y = 240;
		sprites[i].dx = 0;  
		sprites[i].dy = 0; 
		sprites[i].hit = 1;
		sprites[i].index = i;
	}
	sprites[20].id = 10; //0
	sprites[21].id = 10; //0
	sprites[22].id = 10; //0
	for (int i = 20; i < 23; i++) {
		sprites[i].x = 480+32+32*(i-20); 
		sprites[i].y = 290;
		sprites[i].dx = 0;  
		sprites[i].dy = 0; 
		sprites[i].hit = 1;
		sprites[i].index = i;
	}
}

void update_combo(sprite *sprites, const int combo) {
	int huds = (int)combo/100;
	int tens = (int)(combo - huds*100)/10;
	int ones = combo - huds*100 - tens*10;
	if (huds == 0) huds = 10;
	if (tens == 0) tens = 10;
	if (ones == 0) ones = 10;
	sprites[14].id = huds; //100s	
	sprites[15].id = tens; //10s
	sprites[16].id = ones; //1s
	return;
}

void update_score(sprite *sprites, const int score) {
	int huds = (int)score/100;
	int tens = (int)(score - huds*100)/10;
	int ones = score - huds*100 - tens*10;
	if (huds == 0) huds = 10;
	if (tens == 0) tens = 10;
	if (ones == 0) ones = 10;
	sprites[6].id = huds; //100s	
	sprites[7].id = tens; //10s
	sprites[8].id = ones; //1s
	return;
}

void update_max(sprite *sprites, const int max) {
	int huds = (int)max/100;
	int tens = (int)(max - huds*100)/10;
	int ones = max - huds*100 - tens*10;
	if (huds == 0) huds = 10;
	if (tens == 0) tens = 10;
	if (ones == 0) ones = 10;
	sprites[20].id = huds; //100s	
	sprites[21].id = tens; //10s
	sprites[22].id = ones; //1s
	return;
}
// dedicate all sprites below 

// spawns a block with sprite.id depending on note
void spawnnote(sprite* sprites, int note) {
	//scans sprite array for empty sprite
	if (note == 0) return;
	int i, j;
	for (i = 23; i < SIZE; i++) {
	    if (sprites[i].id == 0) break;
	}
	for (j = i+1; j < SIZE; j++) {
	    if (sprites[j].id == 0) break;
	}
	// change sprite information to match note
	sprites[i].x = 28 + 120*(note-1); 
	sprites[i].y = 0;
	sprites[i].dx = 0;  
	sprites[i].dy = 1; 
	sprites[i].id = note*2 + 16;
	sprites[i].hit = 0;
	sprites[i].index = i;
	sprites[j].x = 60 + 120*(note-1); 
	sprites[j].y = 0;
	sprites[j].dx = 0;  
	sprites[j].dy = 1; 
	sprites[j].id = note*2 + 17;
	sprites[j].hit = 0;
	sprites[j].index = j;
	return;
}


// return sprite id in region, -1 if none
int check_valid_region(sprite* sprites, int start) {
    int i;
    //int cf = *combo_flag;
    for (i = start; i < SIZE; i++) {
	    if ((sprites[i].y > 330) && (sprites[i].y < 410) && (sprites[i].id != 0) && (sprites[i].hit == 0)) {
	        //if (sprites[i].y == 480) cf = 0;
	        return i;
	    }
	}
	return -1;
}

void screen_refresh(sprite* sprites) {
    for (int i = 1; i < SIZE; i++) {
		sprites[i].x = 630; 
		sprites[i].y = 470;
		sprites[i].dx = 0;  
		sprites[i].dy = 0; 
		sprites[i].id = 0;
		sprites[i].hit = 1;
		sprites[i].index = i;
	}
	return;
}

// simple game of hitting random falling notes when they reach the green zone
int main()
{
    
	vga_zylo_arg_t vzat;

	aud_arg_t aat;
	aud_mem_t amt;
	vga_pixel_axis_t position;
	vga_pixel_arg_t vla;
	int i;
	int** pixel_values;
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
	
	sprite *sprites = NULL;	
	
	int score = 0;
	int combo = 0;
	//packet of sprite data to send
	vga_zylo_data_t vzdt;
	
	int combo_flag = 1; 
	int counter = 0; 	
	int gamecounter = 0;
    int validleft, validright;
    int max = 0;
    int hitcount = 0;
	int MAX_NOTE_COUNT = 500 * 512;
	int posx,posy;    
	FILE *file;

	file = fopen("/root/cp/test/1.txt", "w");
	printf("start.");
	pixel_values = (int **)malloc(155* sizeof(int *));
	for (int i = 0; i < 150; i++) pixel_values[i] = (int *)malloc(515 * sizeof(int));
	while (counter < MAX_NOTE_COUNT + 5) {
        	amt.data = get_aud_data(aud_fd);
		//if (amt.data == 0) continue;
        	fprintf(file,"%d\n",amt.data);
		//pause to let hardware catch up
		int x;
		x = amt.data/700;
        	printf("%d__%d\n", x,counter);
		if (x > 150) x = 150;
		if (x < 0) x = 0;
		for (int i = 0; i < x; i++){
			pixel_values[i][counter % 512] = 255;
		}
		for (int j = x; j < 150; j++){
			pixel_values[j][counter % 512] = 1;	
		}
		counter++;
		if (counter % 512 == 0 && counter != 0){
			for (int i = 0 ; i < 150 ; i++) {
				position.y1_axis = i >> 8;
				position.y2_axis = i % 256;
				for (int j = 0; j < 512; j++) {
					vga_pixel_color_t tmp;
					tmp.red = pixel_values[i][j];
					position.x1_axis = j >> 8;
					position.x2_axis = j % 256;
					set_background_color(&tmp);
					set_pixel_axis(&position);
				}
			}
		}
		usleep(800);
	}
	fflush(file);
    fclose(file);
	free (sprites);
	return 0;
}
