#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#ifndef C_AI
#define C_AI

// typedef struct Neuron Neuron;
typedef struct Network Network;
typedef struct Layer Layer;

struct Layer {
  double * neuron_values;
  double * neuron_biases;
  int num_neurons;
  double *weights;

  Layer *output_layer;
  int num_outputs;
};

struct Network {
  Layer *layers;
  int num_layers;
  int num_hidden_layers;

  Layer *inputs;
  int num_inputs;

  Layer *outputs;
  int num_outputs;
};

int initialise_network(Network *network, int num_inputs, int num_outputs, int num_hidden_layers, int hidden_layer_length);
void initialise_layer(Layer *layer, int num_neurons, Layer *output_layer);
void free_layer(Layer *layer);
void free_network(Network *network);

void feed_forward(Layer *layer);
void evaluate_network(Network *network);


// -=-==-=-==-=-=-=-=-=-=--==--= DEBUGGING FUNCTIONS
void print_output_layer_values(Network *network);


#endif

