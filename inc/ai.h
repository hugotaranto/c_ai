#ifndef C_AI
#define C_AI

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

// typedef struct Neuron Neuron;
typedef struct Network Network;
typedef struct Layer Layer;
typedef struct CostMap CostMap;
typedef struct CostMapLayer CostMapLayer;
typedef struct NetworkTestData NetworkTestData;

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

struct CostMap {
  int num_iterations;
  double cumulative_cost;

  int num_layers;
  int num_inputs;
  int num_outputs;
  int hidden_layer_length;

  CostMapLayer *layers;

};

struct CostMapLayer {
  int num_neurons;
  int num_outputs;
  double *weights;
  double **ordered_weights;
  double *biases;

  // need to keep track of the partial derivatives
  double *cost_derivative_of_values;
};

struct NetworkTestData {
  int num_data_points;
  int input_length;
  double **inputs;

  int output_length;
  double **outputs;
};

//------------------------------------------------------------------------------------------------------------------------
// NetWork Functions
int initialise_network(Network *network, int num_inputs, int num_outputs, int num_hidden_layers, int hidden_layer_length);
void initialise_layer(Layer *layer, int num_neurons, Layer *input_layer, Layer *output_layer);
void free_layer(Layer *layer);
void free_network(Network *network);

void feed_forward(Layer *layer);
void evaluate_network(Network *network);

//------------------------------------------------------------------------------------------------------------------------
// Training Functions
double get_cost_funtion(Network *network, double *expected);
void back_propogate(Network *network, double *expected, CostMap *costmap);

void initialise_cost_map(CostMap *costmap, Network* network);
void free_cost_map(CostMap *costmap);
void reset_cost_map(CostMap *costmap);
void initialise_cost_map_layer(CostMapLayer *layer, int num_neurons, int num_outputs);
void free_cost_map_layer(CostMapLayer *layer);

int apply_cost_map(Network *network, CostMap *costmap, double learning_rate);

void gradient_descent_train(Network *network, double **inputs, double **expected_outputs, int num_inputs, double learning_rate);
int stochastic_gradient_descent_train(Network *network, NetworkTestData *test_data, int batch_size, int epoch_count, double learning_rate);

void swap(int *a, int *b);
void shuffle_indices(int *indices, int num_indices);

//------------------------------------------------------------------------------------------------------------------------
// Global Functions
void save_network_to_file(Network *network);
Network* load_network_from_file();


// -=-==-=-==-=-=-=-=-=-=--==--= DEBUGGING FUNCTIONS
void print_output_layer_values(Network *network);


#endif

