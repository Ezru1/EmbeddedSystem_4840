#ifndef _AUD_H
#define _AUD_H

#include <linux/ioctl.h>

// recieve
// typedef struct {
// 	int data;
// } aud_amp_t;

typedef struct {
	int data;
	int address;
	int size;
        int mode;
} aud_mem_t;

typedef struct {
//	aud_amp_t audio;
	aud_mem_t memory;
} aud_arg_t;

typedef struct {
  int limit;
  int address;
  int data;
  int mode;
} mem;

#define AUD_MAGIC 'q'

/* ioctls and their arguments */
#define AUD_READ_DATA  	    _IOR(AUD_MAGIC, 1, aud_arg_t *)
#endif
