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
    print_image(image_data->ordered_data[i++]);
  }
}

ImageData* MNIST_load_image_data() {

  char filename[256];
  printf("Enter path to MNIST image data. (e.g. ../data/): ");
  scanf("%255s", filename);  // avoid buffer overflow

  // Combine with current directory (optional)
  char fullpath[512];
  snprintf(fullpath, sizeof(fullpath), "./%strain-images.idx3-ubyte", filename); // saves in current folder


  FILE *f = fopen(fullpath, "rb");
  if (!f) {
    perror("Failed to open image file");
    return NULL;
  }

  // Read header
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
    fprintf(stderr, "Invalid magic number: %u\n", magic_number);
    fclose(f);
    return NULL;
  }

  if (num_rows != IMAGE_ROWS || num_cols != IMAGE_COLS) {
    fprintf(stderr, "Unexpected image size: %ux%u\n", num_rows, num_cols);
    fclose(f);
    return NULL;
  }

  printf("Loading %u MNIST images into memory...\n", num_images);

  ImageData *image_data = malloc(sizeof(ImageData)); 
  image_data->num_images = num_images;

  // Allocate a big chunk of memory
  size_t total_size = (size_t)num_images * IMAGE_SIZE;
  image_data->data = malloc(total_size);
  uint8_t *images = image_data->data;

  if (!images) {
    perror("Failed to allocate memory");
    fclose(f);
    free(image_data);
    return NULL;
  }

  // Read all images at once
  size_t read = fread(images, 1, total_size, f);
  fclose(f);

  if (read != total_size) {
    fprintf(stderr, "Only read %zu bytes, expected %zu\n", read, total_size);
    free(images);
    free(image_data);
    return NULL;
  }

  image_data->ordered_data = malloc(sizeof(uint8_t*) * num_images);
  uint8_t **ordered_images = image_data->ordered_data;

  for(int i = 0; i < num_images; i++) {
    ordered_images[i] = &images[i * 784];
  }

  return image_data;
}


void free_image_data(ImageData *image_data) {
  free(image_data->data);
  free(image_data->ordered_data);
  free(image_data);
}



