#include "ai.h"
#include <cblas.h>

int main() {

  Network *test_network = malloc(sizeof(Network));
  initialise_network(test_network, 784, 10, 2, 16);

  printf("Network created with %d connections!\n", test_network->num_connections);

  free_network(test_network);

}

