#include "ai.h"
#include "MNIST.h"
#include "xor.h"


// int main() {
//   Network *xor_network = train_xor_network();
//   test_xor_outputs(xor_network);
//   free_network(xor_network);
// }


int main() {
  ImageData *image_data;
  image_data = MNIST_load_image_data();

  print_all_images(image_data);

  free_image_data(image_data);
}



