/*
 * Userspace program that communicates with the aud and vga_vylo device driver
 * through ioctls
 * current amplitude will be represented as the y position of the ball from vga_vylo
 * reads audio and then sends amplituded
 * ayu2126
 * Columbia University
 */

#include <stdio.h>
#include "interfaces.h"
#include "vga_zylo.h"
#include "aud.h"
#include <sys/ioctl.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <string.h>
#include <unistd.h>
#include <stdlib.h>

#define X_MAX 640 
#define Y_MAX 479

int vga_zylo_fd;
int aud_fd;

void updateBall(sprite *obj) {
	obj->x += obj->dx;
	obj->y += obj->dy;
	
	if (obj->x < 0 || obj->x >= X_MAX)
		obj->dx = -obj->dx;

	if (obj->y < 0 || obj->y >= Y_MAX) {
		obj->dy = -obj->dy;
		obj->id = obj->baseid;
	}
	// if () {
	// 	obj->dy = -obj->dy;
	// 	obj->id = 0;
	// }
}

int main()
{
	vga_zylo_arg_t vzat;
	
	aud_arg_t aat;
	aud_mem_t amt;

	sprite *sprites = NULL;	
	int r = rand() % 20;
	sprites = calloc(SIZE, sizeof(*sprites));
	//index 0 acts strangly
	//'SCORE'
	sprites[1].baseid = 17; //S
	sprites[2].baseid = 12; //C
	sprites[3].baseid = 15; //O
	sprites[4].baseid = 16; //R
	sprites[5].baseid = 13; //E
	for (int i = 1; i < 6; i++) {
		sprites[i].x = 480+32*(i-1); 
		sprites[i].y = 40;
		sprites[i].dx = 0;  
		sprites[i].dy = 0; 
		sprites[i].id = sprites[i].baseid;
		sprites[i].index = i;
	}
	sprites[6].baseid = 10; //0
	sprites[7].baseid = 10; //0
	sprites[8].baseid = 10; //0
	for (int i = 6; i < 9; i++) {
		sprites[i].x = 480+32+32*(i-6); 
		sprites[i].y = 90;
		sprites[i].dx = 0;  
		sprites[i].dy = 0; 
		sprites[i].id = sprites[i].baseid;
		sprites[i].index = i;
	}

	//'COMBO'
	sprites[9].baseid =  12; //C
	sprites[10].baseid = 15; //O
	sprites[11].baseid = 14; //M
	sprites[12].baseid = 11; //B
	sprites[13].baseid = 15; //O
	for (int i = 9; i < 14; i++) {
		sprites[i].x = 480+32*(i-9); 
		sprites[i].y = 140;
		sprites[i].dx = 0;  
		sprites[i].dy = 0; 
		sprites[i].id = sprites[i].baseid;
		sprites[i].index = i;
	}
	sprites[14].baseid = 10; //0
	sprites[15].baseid = 10; //0
	sprites[16].baseid = 10; //0
	for (int i = 14; i < 17; i++) {
		sprites[i].x = 480+32+32*(i-14); 
		sprites[i].y = 190;
		sprites[i].dx = 0;  
		sprites[i].dy = 0; 
		sprites[i].id = sprites[i].baseid;
		sprites[i].index = i;
	}

	// bouncing notes (for testing)
	sprites[17].baseid = 20; //ol
	sprites[18].baseid = 21; //or
	for (int i = 17; i < 19; i++) {
		sprites[i].x = 32 + 32*(i-17); 
		sprites[i].y = 10;
		sprites[i].dx = 0;  
		sprites[i].dy = 1; 
		sprites[i].id = sprites[i].baseid;
		sprites[i].index = i;
	}
	sprites[19].baseid = 18; //bl
	sprites[20].baseid = 19; //br
	for (int i = 19; i < 21; i++) {
		sprites[i].x = 32 + 96 + 32*(i-19); 
		sprites[i].y = 40;
		sprites[i].dx = 0;  
		sprites[i].dy = 1; 
		sprites[i].id = sprites[i].baseid;
		sprites[i].index = i;
	}
	sprites[21].baseid = 22; //pl
	sprites[22].baseid = 23; //pr
	for (int i = 21; i < 23; i++) {
		sprites[i].x = 32 + 2 * 96 + 32*(i-21);  
		sprites[i].y = 70;
		sprites[i].dx = 0;  
		sprites[i].dy = 1; 
		sprites[i].id = sprites[i].baseid;
		sprites[i].index = i;
	}
	sprites[23].baseid = 24; //pl
	sprites[24].baseid = 25; //pr
	for (int i = 23; i < 25; i++) {
		sprites[i].x = 32 + 3*96 + 32*(i-23);  
		sprites[i].y = 100;
		sprites[i].dx = 0;  
		sprites[i].dy = 1; 
		sprites[i].id = sprites[i].baseid;
		sprites[i].index = i;
	}
	sprites[25].baseid = 20; //ol
	sprites[26].baseid = 21; //or
	for (int i = 25; i < 27; i++) {
		sprites[i].x = 32 + 32*(i-25); 
		sprites[i].y = 130;
		sprites[i].dx = 0;  
		sprites[i].dy = 1; 
		sprites[i].id = sprites[i].baseid;
		sprites[i].index = i;
	}
	sprites[27].baseid = 18; //bl
	sprites[28].baseid = 19; //br
	for (int i = 27; i < 21; i++) {
		sprites[i].x = 32 + 96 + 32*(i-27); 
		sprites[i].y = 160;
		sprites[i].dx = 0;  
		sprites[i].dy = 1; 
		sprites[i].id = sprites[i].baseid;
		sprites[i].index = i;
	}
	sprites[29].baseid = 22; //pl
	sprites[30].baseid = 23; //pr
	for (int i = 29; i < 23; i++) {
		sprites[i].x = 32 + 2*96 + 32*(i-29);  
		sprites[i].y = 190;
		sprites[i].dx = 0;  
		sprites[i].dy = 1; 
		sprites[i].id = sprites[i].baseid;
		sprites[i].index = i;
	}
	sprites[31].baseid = 24; //pl
	sprites[32].baseid = 25; //pr
	for (int i = 31; i < 33; i++) {
		sprites[i].x = 32 + 3*96 + 32*(i-31);  
		sprites[i].y = 220;
		sprites[i].dx = 0;  
		sprites[i].dy = 1; 
		sprites[i].id = sprites[i].baseid;
		sprites[i].index = i;
	}
	
	// empty sprites
	// for (int i = 33; i < SIZE; i++) {
	// 	sprites[i].x = 0;
	//	sprites[i].y = 0;
	// 	sprites[i].dx = 0;  
	// 	sprites[i].dy = 0; 
	// 	sprites[i].baseid = 0;
	// 	sprites[i].id = sprites[i].baseid;
	// 	sprites[i].index = i;
	// }

	// /mem mem_obj = {.data = 0, .address = 0, .limit = 48000, .mode = 1};

	static const char filename1[] = "/dev/vga_zylo";
	static const char filename2[] = "/dev/aud";

	printf("VGA zylo test Userspace program started\n");
	printf("%d\n", sizeof(int));	
	printf("%d\n", sizeof(short));

	if ((vga_zylo_fd = open(filename1, O_RDWR)) == -1) {
		fprintf(stderr, "could not open %s\n", filename1);
		return -1;
	}
	if ((aud_fd = open(filename2, O_RDWR)) == -1) {
		fprintf(stderr, "could not open %s\n", filename2);
		return -1;
	}
 	FILE *fp = fopen("test.txt", "w");
	if (fp == NULL)	return -1;

	vga_zylo_data_t vzdt;
 	
	while (s) {
		//package the sprites together	
		for (int i = 0; i < SIZE; i++) {
			vzdt.data[i] = (sprites[i].index<<26) + (sprites[i].id<<20) + (sprites[i].y<<10) + (sprites[i].x<<0);
		}
		//send package to hardware
		send_sprite_positions(&vzdt);
		//update spirtes on software side
		for (int i = 0; i < SIZE; i++) {
			updateBall(&sprites[i]);
		}
		//pause to let hardware catch up
		usleep(10000);
	}
	free (sprites);

	// amt.size = (int) mem_obj.limit;
	// amt.mode = (int) mem_obj.mode;
	// usleep(1200000);
	// send_mode(&amt);
	// send_limit(&amt);
	// usleep(12000000);
	// int amp = 0;
	// amt.size = 0;
	// // send_limit(&amt);
	// for(int counter = 0; counter < amt.size; counter++) {
	// 		mem_obj.address = counter;
	// 		amt.address = mem_obj.address;
	// 	//printf("%d: ", amt.address);
	// 		send_address(&amt);	
	// 		amt.data = get_data();
	// 		amp = amt.data;
	// 	//printf("%d\n", amt.data);
	// 		fprintf(fp, "%08x\n", amp);
	// }
	// fclose(fp);
	

	return 0;
}
