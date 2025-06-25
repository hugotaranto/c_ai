#include "ai.h"
#include "MNIST.h"
#include <time.h>

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

  // make the network
  // then train it

  Network *network = malloc(sizeof(Network));
  // initialise_network(Network *network, int num_inputs, int num_outputs, int num_hidden_layers, int hidden_layer_length)
  initialise_network(network, training_data->input_length, training_data->output_length, 2, 16);

  // stochastic_gradient_descent_train(Network *network, NetworkTrainingData *training_data, int batch_size, int epoch_count, double learning_rate)
  clock_t start = clock();
  stochastic_gradient_descent_train(network, training_data, 100, 30, 0.1);
  clock_t end = clock();
  double time_taken = (double)(end - start) / CLOCKS_PER_SEC;
  printf("time taken to train: %f\n", time_taken);

  save_network_to_file(network);

  // Network *network = load_network_from_file();

  // then test the network
  MNIST_test_network(network, training_data, image_data);

  free_network(network);
  free_image_data(image_data);
}
//
// int main() {
//
//   Network *xor_network = malloc(sizeof(Network));
//   // initialise_network(Network *network, int num_inputs, int num_outputs, int num_hidden_layers, int hidden_layer_length)
//   initialise_network(xor_network, 2, 2, 1, 2);
//
//   // create the dataset required. In this case we can just create all cases of a xor gate
//   double **inputs = malloc(sizeof(double*) * 4);
//   double **expected_outputs = malloc(sizeof(double*) * 4);
//
//   for(int i = 0; i < 4; i++) {
//     inputs[i] = malloc(sizeof(double) * 2);
//     expected_outputs[i] = malloc(sizeof(double) * 2);
//   }
//
//   inputs[0][0] = 0; inputs[0][1] = 0;
//   inputs[1][0] = 0; inputs[1][1] = 1;
//   inputs[2][0] = 1; inputs[2][1] = 0;
//   inputs[3][0] = 1; inputs[3][1] = 1;
//
//   // outputs are [0] for 0, [1] for 1
//
//   expected_outputs[0][0] = 1; expected_outputs[0][1] = 0;
//   expected_outputs[1][0] = 0; expected_outputs[1][1] = 1;
//   expected_outputs[2][0] = 0; expected_outputs[2][1] = 1;
//   expected_outputs[3][0] = 1; expected_outputs[3][1] = 0;
//
//   
//   // make the training data
//   NetworkTrainingData *data = malloc(sizeof(NetworkTrainingData));
//   data->num_data_points = 4;
//   data->input_length = 2;
//   data->output_length = 2;
//   data->inputs = inputs;
//   data->outputs = expected_outputs;
//
//
//   clock_t start = clock();
//   // for(int i = 0; i < 100000; i++) {
//   //   gradient_descent_train(xor_network, inputs, expected_outputs, 4, 0.1);
//   // }
//   stochastic_gradient_descent_train(xor_network, data, 4, 100000, 0.1);
//   clock_t end = clock();
//
//   printf("time to train: %f\n", ((double) (end - start)) / CLOCKS_PER_SEC);
//
//   // test_outputs(xor_network);
//
//   // save_network_to_file(xor_network);
//   // Network *copy;
//   // copy = load_network_from_file();
//
//   // test_outputs(copy);
//
//   free_network(xor_network);
//   // free_network(copy);
// }
