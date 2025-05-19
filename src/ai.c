#include "ai.h"
#include <cblas.h>

int initialise_network(Network *network, int num_inputs, int num_outputs, int num_hidden_layers, int hidden_layer_length) {

  if (num_inputs <= 0 || num_outputs <=0 || num_hidden_layers <=0 || hidden_layer_length <= 0) {
    return -1;
  }

  network->num_layers = num_hidden_layers + 2;
  network->num_hidden_layers = num_hidden_layers;
  network->num_inputs = num_inputs;
  network->num_outputs = num_outputs;

  network->layers = malloc(sizeof(Layer) * network->num_layers); // allocate the hidden layers and input/output
  network->inputs = &network->layers[0];
  network->outputs = &network->layers[network->num_layers - 1];

  // now need to allocate each layer
  // first do the outputs
  initialise_layer(network->outputs, network->num_outputs, NULL);

  // then do the hidden layers
  for(int i = network->num_layers; i > 0; i--) {
    initialise_layer(&network->layers[i], hidden_layer_length, &network->layers[i + 1]);
  }

  // then the input layer
  initialise_layer(network->inputs, network->num_inputs, &network->layers[1]);

  return 0;
}


void initialise_layer(Layer *layer, int num_neurons, Layer *output_layer) {

  layer->num_neurons = num_neurons;
  layer->neuron_values = malloc(sizeof(double) * layer->num_neurons);
  layer->neuron_biases = malloc(sizeof(double) * layer->num_neurons);
  layer->output_layer = output_layer;

  memset(layer->neuron_values, 0, sizeof(double) * layer->num_neurons);
  memset(layer->neuron_biases, 0, sizeof(double) * layer->num_neurons);

  if (output_layer != NULL) {
    layer->num_outputs = output_layer->num_neurons;
    layer->weights = malloc(sizeof(double) * layer->num_outputs * layer->num_neurons);

  } else {
    layer->weights = NULL;
    layer->num_outputs = 0;
  }
}

void free_layer(Layer *layer) {
  free(layer->neuron_values);
  free(layer->neuron_biases);
  if(layer->weights != NULL) {
    free(layer->weights);
  }
}

void free_network(Network *network) {
  for(int i = 0; i < network->num_layers; i++) {
    free_layer(&network->layers[i]);
  }
  free(network->layers);
  free(network);
}

void feed_forward(Layer *layer) {

  // do a matrix multiplication using cblas (fast)
  int rows = layer->num_outputs; // M
  int columns = layer->num_neurons; // K

  double *output_matrix;
  output_matrix = malloc(sizeof(double) * layer->num_outputs);
  
  cblas_dgemm(
    CblasColMajor, CblasNoTrans, CblasNoTrans,
    rows, 1, columns, // M, N, K
    1.0,
    layer->weights, rows,
    layer->output_layer->neuron_values, columns,
    0.0,
    output_matrix, rows
  );

  // then add the bias and update the output values
  for(int i = 0; i < layer->num_outputs; i++) {
    layer->output_layer->neuron_values[i] = 1 / (1 + exp(-output_matrix[i] + layer->output_layer->neuron_biases[i]));
  }

  free(output_matrix);
}

void evaluate_network(Network *network) {
  for(int i = 0; i < network->num_layers - 1; i++) {
    feed_forward(&network->layers[i]);
  }
}



// -=-==-=-==-=-=-=-=-=-=--==--= DEBUGGING FUNCTIONS
void print_output_layer_values(Network *network) {
  printf("[");
  for(int i = 0; i < network->num_outputs - 1; i++) {
    printf(" %f, ", network->outputs->neuron_values[i]);
  }
  printf(" %f]\n", network->outputs->neuron_values[network->num_outputs - 1]);
}

