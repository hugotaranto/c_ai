#ifndef C_AI_NETWORK_H
#define C_AI_NETWORK_H
#include "ProjectForwards.h"
#include "UtilAi.h"

typedef struct Layer Layer;

struct Network {
  Layer *layers;
  int num_layers;
  int num_hidden_layers;
  int hidden_layer_length;

  Layer *inputs;
  int num_inputs;

  Layer *outputs;
  int num_outputs;
};

struct Layer {
  double * neuron_values;
  double * neuron_biases;
  int num_neurons;
  double *weights;

  Layer *output_layer;
  int num_outputs;

  Layer *input_layer;
  int num_inputs;

  double **ordered_weights; // [num_outputs][num_neurons] -- weights stored for the next layer
};


//------------------------------------------------------------------------------------------------------------------------
// NetWork Functions
//------------------------------------------------------------------------------------------------------------------------
int initialise_network(Network *network, int num_inputs, int num_outputs, int num_hidden_layers, int hidden_layer_length);
void free_network(Network *network);
void evaluate_network(Network *network);

void save_network_to_file(Network *network);
Network* load_network_from_file();

//------------------------------------------------------------------------------------------------------------------------
// Layer Functions
//------------------------------------------------------------------------------------------------------------------------
void initialise_layer(Layer *layer, int num_neurons, Layer *input_layer, Layer *output_layer);
void free_layer(Layer *layer);
void feed_forward(Layer *layer);

//------------------------------------------------------------------------------------------------------------------------
// Debugging Functions
//------------------------------------------------------------------------------------------------------------------------
void print_output_layer_values(Network *network);

#endif
