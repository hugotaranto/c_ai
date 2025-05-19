#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#ifndef AI_STUFF
#define AI_STUFF

typedef struct Neuron Neuron;
typedef struct Network Network;

struct Neuron {

  double value;
  double bias;

  Neuron *input_neurons;
  double *input_weights; // if wanted to change double data type, need to change neuron initialisation function
  int num_inputs;

};

struct Network {
  // a network has an input layer
  Neuron *inputs;
  int num_inputs;
  // and an output layer
  Neuron *outputs;
  int num_outputs;

  // and hidden layers
  Neuron **hidden_layers;
  int num_hidden_layers;
  int hidden_layer_length;

  int num_connections;

};

void initialise_neuron(Neuron *neuron, int num_inputs, Neuron *input_neurons);
int initialise_network(Network *network, int num_inputs, int num_outputs, int num_hidden_layers, int length_hidden_layers);
void free_neuron(Neuron *neuron);
void free_network(Network *network);
double get_neuron_value(Neuron *neuron);

#endif

