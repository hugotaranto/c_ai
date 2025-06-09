#ifndef _MNIST_DATA_H
#define _MNIST_DATA_H

#include "global_params.h"
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define IMAGE_ROWS 28
#define IMAGE_COLS 28
#define IMAGE_SIZE (IMAGE_ROWS * IMAGE_COLS)

typedef struct ImageData ImageData;
struct ImageData {
  int num_images;

  uint8_t *data;
  uint8_t **ordered_data;
};

ImageData* MNIST_load_image_data();
void print_image(uint8_t *image);
void print_all_images(ImageData *image_data);

void free_image_data(ImageData *image_data);

#endif
