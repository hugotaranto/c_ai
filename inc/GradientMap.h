#ifndef C_AI_GRADIENTMAP_H
#define C_AI_GRADIENTMAP_H
#include "ProjectForwards.h"

// struct Network;
typedef struct GradientMapLayer GradientMapLayer;

struct GradientMap {
  int num_iterations;
  double cumulative_cost;

  int num_layers;
  int num_inputs;
  int num_outputs;
  int hidden_layer_length;

  GradientMapLayer *layers;

};

struct GradientMapLayer {
  int num_neurons;
  int num_outputs;
  double *weights;
  double **ordered_weights;
  double *biases;

  // need to keep track of the partial derivatives
  double *cost_derivative_of_values;
};

//------------------------------------------------------------------------------------------------------------------------
// Gradient Map Functions
//------------------------------------------------------------------------------------------------------------------------
void initialise_gradient_map(GradientMap *gradientmap, Network* network);
void free_gradient_map(GradientMap *gradientmap);
void reset_gradient_map(GradientMap *gradientmap);
int apply_gradient_map(Network *network, GradientMap *gradientmap, double learning_rate);

void initialise_gradient_map_layer(GradientMapLayer *layer, int num_neurons, int num_outputs);
void free_gradient_map_layer(GradientMapLayer *layer);



#endif
