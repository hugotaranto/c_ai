#include "MNIST.h"

// ASCII grayscale mapping (optional)
char grayscale[] = " .:-=+*#%@";

void print_image(uint8_t *image) {
  for (int i = 0; i < IMAGE_ROWS; i++) {
    for (int j = 0; j < IMAGE_COLS; j++) {
      uint8_t pixel = image[i * IMAGE_COLS + j];
      char c = grayscale[pixel / 26];  // 256/10 = ~26
      printf("%c", c);
    }
    printf("\n");
  }
}

void print_all_images(ImageData *image_data) {

  int i = 0;
  while(i < image_data->num_images - 1) {
    char line[MAX_LINE];
    printf("'Enter' to print the next image. type in 'exit' to quit.\n");
    if (!fgets(line, sizeof(line), stdin)) {
      break;
    }
    if (strncmp(line, "exit", 4) == 0) {
      break;
    }
    printf("Label For Below Image: %u\n", image_data->labels[i]);
    print_image(image_data->ordered_data[i++]);
  }
}

ImageData* MNIST_load_image_data() {

  char filename[256];
  printf("Enter path to MNIST image data. (e.g. ../data/): ");
  scanf("%255s", filename);  // avoid buffer overflow

  // Combine with current directory (optional)
  char fullpath[512];
  char label_path[512];
  snprintf(fullpath, sizeof(fullpath), "./%strain-images.idx3-ubyte", filename);
  snprintf(label_path, sizeof(label_path), "./%strain-labels.idx1-ubyte", filename);

  FILE *f = fopen(fullpath, "rb");
  if (!f) {
    perror("Failed to open image file");
    return NULL;
  }

  // Read header of image file
  uint32_t magic_number, num_images, num_rows, num_cols;
  fread(&magic_number, sizeof(uint32_t), 1, f);
  fread(&num_images, sizeof(uint32_t), 1, f);
  fread(&num_rows, sizeof(uint32_t), 1, f);
  fread(&num_cols, sizeof(uint32_t), 1, f);

  // Convert big-endian to host byte order
  magic_number = __builtin_bswap32(magic_number);
  num_images = __builtin_bswap32(num_images);
  num_rows = __builtin_bswap32(num_rows);
  num_cols = __builtin_bswap32(num_cols);

  if (magic_number != 2051) {
    fprintf(stderr, "Invalid magic number in image file: %u\n", magic_number);
    fclose(f);
    return NULL;
  }

  if (num_rows != IMAGE_ROWS || num_cols != IMAGE_COLS) {
    fprintf(stderr, "Unexpected image size: %ux%u\n", num_rows, num_cols);
    fclose(f);
    return NULL;
  }

  // open the label data
  FILE *label_file = fopen(label_path, "rb");
  if (!label_file) {
    perror("Failed to open label file");
    return NULL;
  }

  // Read header of label file
  uint32_t num_labels;
  fread(&magic_number, sizeof(uint32_t), 1, label_file);
  fread(&num_labels, sizeof(uint32_t), 1, label_file);

  magic_number = __builtin_bswap32(magic_number);
  num_labels = __builtin_bswap32(num_labels);

  printf("Magic number: %u\n", magic_number);

  if (magic_number != 2049) {
    fprintf(stderr, "Invalid magic number in label file: %u\n", magic_number);
    fclose(f);
    fclose(label_file);
    return NULL;
  }

  if (num_labels != num_images) {
    perror("Number of labels and images does not match");
    fclose(f);
    fclose(label_file);
    return NULL;
  }

  ImageData *image_data = malloc(sizeof(ImageData)); 
  image_data->num_images = num_images;

  if (!image_data) {
    perror("Failed to allocate memory for image data struct");
    fclose(f);
    fclose(label_file);
    return NULL;
  }

  printf("Loading %u MNIST labels into memory...\n", num_labels);
  image_data->labels = malloc(sizeof(uint8_t) * num_labels);
  if (!image_data->labels) {
    perror("Failed to allocate memory for labels");
    fclose(f);
    fclose(label_file);
    free(image_data);
    return NULL;
  }

  size_t label_read_size = fread(image_data->labels, 1, sizeof(uint8_t) * num_labels, label_file); 
  fclose(label_file);

  printf("Loading %u MNIST images into memory...\n", num_images);

  // Allocate a big chunk of memory
  size_t total_size = (size_t)num_images * IMAGE_SIZE;
  image_data->data = malloc(total_size);
  uint8_t *images = image_data->data;

  if (!images) {
    perror("Failed to allocate memory");
    fclose(f);
    free(image_data);
    free(image_data->labels);
    return NULL;
  }

  // Read all images at once
  size_t read = fread(images, 1, total_size, f);
  fclose(f);

  if (read != total_size) {
    fprintf(stderr, "Only read %zu bytes, expected %zu\n", read, total_size);
    free(image_data->data);
    free(image_data->labels);
    free(image_data);
    return NULL;
  }

  image_data->ordered_data = malloc(sizeof(uint8_t*) * num_images);
  uint8_t **ordered_images = image_data->ordered_data;

  for(int i = 0; i < num_images; i++) {
    ordered_images[i] = &images[i * IMAGE_SIZE];
  }

  return image_data;
}

void free_image_data(ImageData *image_data) {
  free(image_data->data);
  free(image_data->ordered_data);
  free(image_data);
}

NetworkTrainingData* convert_to_network_data(ImageData *image_data) {

  size_t image_data_size = image_data->num_images * IMAGE_SIZE * (sizeof(double) / sizeof(uint8_t));
  printf("Allocating %lu bytes of image data.\n", image_data_size);

  double **converted_image_data = malloc(sizeof(double*) * image_data->num_images);
  if (!converted_image_data) {
    perror("Could not Malloc converted image data");
    return NULL;
  }
  
  for(int i = 0; i < image_data->num_images; i++) {
    converted_image_data[i] = malloc(sizeof(double) * IMAGE_SIZE);
    if (!converted_image_data[i]) {
      perror("Could not Allocate Image within converted Image data");
      for(int z = i - 1; z >=0; z--) {
        free(converted_image_data[z]);
      }
      free(converted_image_data);
      return NULL;
    }
    for(int j = 0; j < IMAGE_SIZE; j++) {
      converted_image_data[i][j] = (double)image_data->ordered_data[i][j] / 255.0;
    }
  }

  size_t label_data_size = image_data->num_images * 10 * sizeof(double);
  printf("Allocating %lu bytes of image labels.\n", label_data_size);

  double **label_data = malloc(sizeof(double*) * image_data->num_images);
  if (!label_data) {
    perror("Could not Allocate Converted Label Data");
    free_image_training_data(converted_image_data, image_data->num_images);
    return NULL;
  }

  for(int i = 0; i < image_data->num_images; i++) {
    label_data[i] = malloc(sizeof(double) * NUM_OUPUTS);
    if (!label_data[i]) {
      perror("Could not Allocate Output Data Row");
      free_image_training_data(converted_image_data, image_data->num_images);
      free_image_training_data(label_data, i);
      return NULL;
    }
    uint8_t label = image_data->labels[i];
    memset(label_data[i], 0, sizeof(double) * NUM_OUPUTS);
    label_data[i][label] = 1;

  }

  // all of the data has been converted safely
  // now just convert to ai training data struct
  NetworkTrainingData *training_data = malloc(sizeof(NetworkTrainingData));

  training_data->num_data_points = image_data->num_images;
  training_data->input_length = IMAGE_SIZE;
  training_data->output_length = NUM_OUPUTS;
  training_data->inputs = converted_image_data;
  training_data->outputs = label_data;

  return training_data;

}

void free_image_training_data(double **data, int num_data_points) {
  for(int i = 0; i < num_data_points; i++) {
    free(data[i]);
  }
  free(data);
}


void MNIST_test_network(Network *network, NetworkTrainingData *training_data, ImageData *image_data) {

  int i = 0;
  while(i < image_data->num_images - 1) {
    char line[MAX_LINE];
    printf("'Enter' to print the next image. type in 'exit' to quit.\n");
    if (!fgets(line, sizeof(line), stdin)) {
      break;
    }
    if (strncmp(line, "exit", 4) == 0) {
      break;
    }
    printf("Label For Below Image: %u\n", image_data->labels[i]);

    // get the networks prediction
    // feed the data in
    memcpy(network->inputs->neuron_values, training_data->inputs[i], sizeof(double) * network->num_inputs);

    // evaluate the network
    evaluate_network(network);

    double max = network->outputs->neuron_values[0];
    int guess = 0;

    printf("Network Output: [%f", max);
    // get the highest probable output
    for(int j = 1; j < network->num_outputs; j++) {
      printf(", %f", network->outputs->neuron_values[j]);
      if(network->outputs->neuron_values[j] > max) {
        max = network->outputs->neuron_values[j];
        guess = j;
      }
    }
    printf("]\nNetworks Guess: %d\n", guess);

    print_image(image_data->ordered_data[i++]);
  }

}




