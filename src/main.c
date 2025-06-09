#include "ai.h"
#include <time.h>

#define MAX_LINE 100

void test_outputs(Network *network) {

  while (1) {
    char line[MAX_LINE];
    double a, b;

    printf("Enter two numbers (e.g. 1 0) or type 'exit' to quit: ");
    if (!fgets(line, sizeof(line), stdin)) {
        break; // EOF or error
    }

    // Check for "exit"
    if (strncmp(line, "exit", 4) == 0) {
        break;
    }

    // Try to parse two numbers
    if (sscanf(line, "%lf %lf", &a, &b) != 2) {
        printf("Invalid input. Please enter two numbers or 'exit'.\n");
        continue;
    }

    // Run the network
    network->inputs->neuron_values[0] = a;
    network->inputs->neuron_values[1] = b;
    evaluate_network(network);

    if (network->outputs->neuron_values[0] >= network->outputs->neuron_values[1]) {
        printf("output: 0\n");
    } else {
        printf("output: 1\n");
    }
  }
}

int main() {
  Network *xor_network;
  xor_network = load_network_from_file();

  test_outputs(xor_network);

  free_network(xor_network);
}

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
//   clock_t start = clock();
//   for(int i = 0; i < 1000000; i++) {
//     gradient_descent_train(xor_network, inputs, expected_outputs, 4, 0.1);
//   }
//   clock_t end = clock();
//
//   printf("time to train: %f\n", ((double) (end - start)) / CLOCKS_PER_SEC);
//
//   test_outputs(xor_network);
//
//   save_network_to_file(xor_network);
//   Network *copy;
//   copy = load_network_from_file();
//
//   test_outputs(copy);
//
//   free_network(xor_network);
//   free_network(copy);
// }


