#include "ai.h"

int main() {

  // Network *test_network = malloc(sizeof(Network));
  // initialise_network(test_network, 2, 2, 3, 4);
  //
  // print_output_layer_values(test_network);
  // evaluate_network(test_network);
  // print_output_layer_values(test_network);
  //
  // double *expected = malloc(sizeof(double) * 2);
  // expected[0] = 0;
  // expected[1] = 1;
  //
  // back_propogate(test_network, expected);
  //
  // free_network(test_network);

  Network *xor_network = malloc(sizeof(Network));
  // initialise_network(Network *network, int num_inputs, int num_outputs, int num_hidden_layers, int hidden_layer_length)
  initialise_network(xor_network, 2, 1, 1, 2);

  // set the things
  xor_network->inputs->neuron_values[0] = 0.35;
  xor_network->inputs->neuron_values[1] = 0.7;

  xor_network->inputs->ordered_weights[0][0] = 0.2;
  xor_network->inputs->ordered_weights[0][1] = 0.2;
  xor_network->inputs->ordered_weights[1][0] = 0.3;
  xor_network->inputs->ordered_weights[1][1] = 0.3;

  // biases are 0

  xor_network->layers[1].ordered_weights[0][0] = 0.3;
  xor_network->layers[1].ordered_weights[0][1] = 0.9;

  evaluate_network(xor_network);

  printf("%f, %f, %f\n", xor_network->layers[1].neuron_values[0], xor_network->layers[1].neuron_values[1], xor_network->outputs->neuron_values[0]);


  // // create the dataset required. In this case we can just create all cases of a xor gate
  // double **inputs = malloc(sizeof(double*) * 4);
  // double **expected_outputs = malloc(sizeof(double*) * 4);
  //
  // for(int i = 0; i < 4; i++) {
  //   inputs[i] = malloc(sizeof(double) * 2);
  //   expected_outputs[i] = malloc(sizeof(double));
  // }
  //
  // inputs[0][0] = 0; inputs[0][1] = 0;
  // inputs[1][0] = 0; inputs[1][1] = 1;
  // inputs[2][0] = 1; inputs[2][1] = 0;
  // inputs[3][0] = 1; inputs[3][1] = 1;
  //
  // expected_outputs[0][0] = 0;
  // expected_outputs[1][0] = 1;
  // expected_outputs[2][0] = 1;
  // expected_outputs[3][0] = 0;
  //
  // for(int i = 0; i < 1000000; i++) {
  //   gradient_descent_train(xor_network, inputs, expected_outputs, 4);
  // }

  free_network(xor_network);

}

