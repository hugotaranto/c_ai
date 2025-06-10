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

  // print_all_images(image_data);

  NetworkTrainingData *training_data = convert_to_network_data(image_data);
  //
  // // make the network
  // // then train it
  //
  // Network *network = malloc(sizeof(Network));
  // // initialise_network(Network *network, int num_inputs, int num_outputs, int num_hidden_layers, int hidden_layer_length)
  // initialise_network(network, training_data->input_length, training_data->output_length, 2, 16);
  //
  // // stochastic_gradient_descent_train(Network *network, NetworkTrainingData *training_data, int batch_size, int epoch_count, double learning_rate)
  // stochastic_gradient_descent_train(network, training_data, 100, 300, 0.1);
  //
  // save_network_to_file(network);

  Network *network = load_network_from_file();

  // then test the network
  MNIST_test_network(network, training_data, image_data);

  free_image_data(image_data);
}


