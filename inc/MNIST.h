#ifndef _MNIST_DATA_H
#define _MNIST_DATA_H

#include "ai.h"
#include "global_params.h"
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define IMAGE_ROWS 28
#define IMAGE_COLS 28
#define IMAGE_SIZE (IMAGE_ROWS * IMAGE_COLS)
#define NUM_OUPUTS 10

typedef struct ImageData ImageData;
struct ImageData {
  int num_images;

  uint8_t *data;
  uint8_t **ordered_data;

  uint8_t *labels;
};

ImageData* MNIST_load_image_data();
void print_image(uint8_t *image);
void print_all_images(ImageData *image_data);

void free_image_data(ImageData *image_data);

NetworkTrainingData* convert_to_network_data(ImageData *image_data);
void free_image_training_data(double **data, int num_data_points);

void MNIST_test_network(Network *network, NetworkTrainingData *training_data, ImageData *image_data);


#endif
