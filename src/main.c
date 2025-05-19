#include "ai.h"

int main() {

  Network *test_network = malloc(sizeof(Network));
  initialise_network(test_network, 784, 10, 2, 16);

  print_output_layer_values(test_network);
  evaluate_network(test_network);
  print_output_layer_values(test_network);

  free_network(test_network);
}

