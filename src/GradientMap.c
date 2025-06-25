#include "GradientMap.h"

#include "Network.h"

//------------------------------------------------------------------------------------------------------------------------
// Gradient Map Functions
//------------------------------------------------------------------------------------------------------------------------
void initialise_gradient_map(GradientMap *gradientmap, Network *network) {

  gradientmap->num_iterations = 0;
  gradientmap->cumulative_cost = 0;
  gradientmap->num_layers = network->num_layers;
  gradientmap->num_outputs = network->num_outputs;
  gradientmap->num_inputs = network->num_inputs;
  gradientmap->hidden_layer_length = network->hidden_layer_length;

  gradientmap->layers = malloc(sizeof(GradientMapLayer) *gradientmap->num_layers);

  initialise_gradient_map_layer(&gradientmap->layers[0], gradientmap->num_inputs, gradientmap->hidden_layer_length);

  for(int i = 1; i < gradientmap->num_layers - 2; i++) {
    initialise_gradient_map_layer(&gradientmap->layers[i], gradientmap->hidden_layer_length, gradientmap->hidden_layer_length);
  }

  initialise_gradient_map_layer(&gradientmap->layers[gradientmap->num_layers - 2], gradientmap->hidden_layer_length, gradientmap->num_outputs);
  initialise_gradient_map_layer(&gradientmap->layers[gradientmap->num_layers - 1], gradientmap->num_outputs, 0);

}

void free_gradient_map(GradientMap *gradientmap) {
  for(int i = 0; i < gradientmap->num_layers; i++) {
    free_gradient_map_layer(&gradientmap->layers[i]);
  }
  free(gradientmap->layers);
  free(gradientmap);
}

void reset_gradient_map(GradientMap *gradientmap) {
  gradientmap->num_iterations = 0;
  gradientmap->cumulative_cost = 0;

  for(int i = 0; i < gradientmap->num_layers; i++) {
    GradientMapLayer *layer = &gradientmap->layers[i];
    memset(layer->weights, 0, sizeof(double) * layer->num_outputs * layer->num_neurons);
    memset(layer->biases, 0, sizeof(double) * layer->num_neurons);
    memset(layer->cost_derivative_of_values, 0, sizeof(double) * layer->num_neurons);
  }
}

int apply_gradient_map(Network *network, GradientMap *gradientmap, double learning_rate) {
  // here the gradientmap needs to be applied to the network
  if (learning_rate <= 0) {
    return -1;
  }

  // the scalar mult should take into account the numer of iterations taken in back propogation to get the average
  double scalar = -learning_rate / gradientmap->num_iterations; 

  // loop through all of the layers in the network
  for(int i = 0; i < network->num_layers; i++) {
    Layer *network_layer = &network->layers[i];
    GradientMapLayer *gradientmap_layer = &gradientmap->layers[i];

    // perform fast matrix action, all this does is make w_i = w_i - learning_rate * weight_gradients_i
    cblas_daxpy(network_layer->num_outputs * network_layer->num_neurons, 
                scalar,
                gradientmap_layer->weights, 1,
                network_layer->weights, 1 
    );

    // then do the same for the bias
    cblas_daxpy(network_layer->num_neurons,
                scalar,
                gradientmap_layer->biases, 1,
                network_layer->neuron_biases, 1
    );

  }

  return 0;
}

void initialise_gradient_map_layer(GradientMapLayer *layer, int num_neurons, int num_outputs) {

  layer->num_neurons = num_neurons;
  layer->num_outputs = num_outputs;

  if (num_outputs != 0) {

    layer->weights = malloc(sizeof(double) * num_outputs * num_neurons);
    // very important that these values all start as 0
    memset(layer->weights, 0, sizeof(double) * num_outputs * num_neurons);

    layer->ordered_weights = malloc(sizeof(double *) * num_outputs);

    for(int i = 0; i < num_outputs; i++) {
      layer->ordered_weights[i] = &layer->weights[i * num_neurons];
    }

  } else {
    layer->ordered_weights = NULL;
    layer->weights = NULL;
  }

  layer->biases = malloc(sizeof(double) * num_neurons);
  // also very important that these start as 0
  memset(layer->biases, 0, sizeof(double) * num_neurons);
  layer->cost_derivative_of_values = malloc(sizeof(double) * num_neurons);
  memset(layer->cost_derivative_of_values, 0, sizeof(double) * num_neurons);
}

void free_gradient_map_layer(GradientMapLayer *layer) {
  if (layer->ordered_weights != NULL) {
    free(layer->ordered_weights);
    free(layer->weights);
  }
  if (layer->biases != NULL) {
    free(layer->biases);
  }
  if (layer->cost_derivative_of_values != NULL) {
    free(layer->cost_derivative_of_values);
  }
}


