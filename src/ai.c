#include "ai.h"

void initialise_neuron(Neuron *neuron, int num_inputs, Neuron *input_neurons) {

  neuron->value = 0;
  neuron->num_inputs = num_inputs;
  neuron->input_neurons = input_neurons;

  neuron->input_weights = malloc(sizeof(double) * num_inputs);

  // randomise all of the initial weights? time for research
  neuron->bias = 0;
  
  // for now just memset to 0
  memset(neuron->input_weights, 0, sizeof(double) * neuron->num_inputs);

}

int initialise_network(Network *network, int num_inputs, int num_outputs, int num_hidden_layers, int length_hidden_layers) {

  if(num_inputs <= 0 || num_outputs <= 0 || num_hidden_layers <= 0 || length_hidden_layers <= 0) {
    return - 1;
  }
  
  network->num_inputs = num_inputs;
  network->num_outputs = num_outputs;
  network->num_hidden_layers = num_hidden_layers;
  network->hidden_layer_length = length_hidden_layers;
  network->num_connections = 0;

  network->inputs = malloc(sizeof(Neuron) * network->num_inputs);
  network->outputs = malloc(sizeof(Neuron) * network->num_outputs);
  network->hidden_layers = malloc(sizeof(Neuron*) * network->num_hidden_layers);

  for(int i = 0; i < network->num_hidden_layers; i++) {
    network->hidden_layers[i] = malloc(sizeof(Neuron) * network->hidden_layer_length);
  }

  // now all of the neurons need to be initialised

  // input neurons don't need anything fancy done
  for(int i = 0; i < network->num_inputs; i++) {
    network->inputs[i].input_neurons = NULL;
    network->inputs[i].input_weights = NULL;
    network->inputs[i].num_inputs = 0;
    network->inputs[i].value = 0;
  }

  // however, the rest do
  for(int i = 0; i < network->num_outputs; i++) {
    initialise_neuron(&network->outputs[i], network->hidden_layer_length, network->hidden_layers[network->num_hidden_layers - 1]);
    network->num_connections += network->hidden_layer_length;
  }

  // initialise all of the neurons in the hidden layers
  for(int i = 0; i < network->num_hidden_layers; i++) {
    for(int j = 0; j < network->hidden_layer_length; j++) {
      // if we are on the first hidden layer, the input neurons/weights are the input layer
      if(i == 0) {
        initialise_neuron(&network->hidden_layers[i][j], network->num_inputs, network->inputs);
        network->num_connections += network->num_inputs;
      } else { // otherwise they will be another hidden layer
        initialise_neuron(&network->hidden_layers[i][j], network->hidden_layer_length, network->hidden_layers[i - 1]);
        network->num_connections += network->hidden_layer_length;
      }

    }
  }

  return 0;

}

void free_neuron(Neuron *neuron) {

  if(neuron->input_weights != NULL) {
    free(neuron->input_weights);
  }

}

void free_network(Network *network) {

  // free the inputs
  for(int i = 0; i < network->num_inputs; i++) {
    free_neuron(&network->inputs[i]);
  }
  free(network->inputs);

  // free the outputs
  for(int i = 0; i < network->num_outputs; i++) {
    free_neuron(&network->outputs[i]);
  }
  free(network->outputs);

  // free all of the hidden neurons and layers
  for(int i = 0; i < network->num_hidden_layers; i++) {
    for(int j = 0; j < network->hidden_layer_length; j++) {
      free_neuron(&network->hidden_layers[i][j]);
    }
    free(network->hidden_layers[i]);
  }
  free(network->hidden_layers); 

  // finally free the network itself
  free(network);
}

double get_neuron_value(Neuron *neuron) {

  // find the weighted sum of all of the input neurons and weights
  if (neuron->input_neurons == NULL || neuron->input_weights == NULL) {
    // have to raise an error here
    return 0.0;
  }

  double weighted_sum = 0;

  for(int i = 0; i < neuron->num_inputs; i++) {
    weighted_sum += neuron->input_weights[i] * neuron->input_neurons[i].value;
  }

  // then get the sigma of this to scale it between 1 and 0
  double sigma = 1 / (1 + exp(-weighted_sum + neuron->bias));
  return sigma;

};


