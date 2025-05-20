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
  network->hidden_layer_length = hidden_layer_length;

  network->layers = malloc(sizeof(Layer) * network->num_layers); // allocate the hidden layers and input/output
  network->inputs = &network->layers[0];
  network->outputs = &network->layers[network->num_layers - 1];

  // now need to allocate each layer
  // first do the outputs
  initialise_layer(network->outputs, network->num_outputs, &network->layers[network->num_layers - 2], NULL);

  // then do the hidden layers
  for(int i = network->num_layers - 2; i > 0; i--) {
    initialise_layer(&network->layers[i], hidden_layer_length, &network->layers[i - 1], &network->layers[i + 1]);
  }

  // then the input layer
  initialise_layer(network->inputs, network->num_inputs, NULL, &network->layers[1]);

  return 0;
}


void initialise_layer(Layer *layer, int num_neurons, Layer *input_layer, Layer *output_layer) {

  layer->num_neurons = num_neurons;
  layer->neuron_values = malloc(sizeof(double) * layer->num_neurons);
  layer->neuron_biases = malloc(sizeof(double) * layer->num_neurons);
  layer->output_layer = output_layer;
  layer->input_layer = input_layer;

  memset(layer->neuron_values, 0, sizeof(double) * layer->num_neurons);
  memset(layer->neuron_biases, 0, sizeof(double) * layer->num_neurons);

  if (output_layer != NULL) {
    layer->num_outputs = output_layer->num_neurons;
    layer->weights = malloc(sizeof(double) * layer->num_outputs * layer->num_neurons);
    layer->ordered_weights = malloc(sizeof(double*) * layer->num_outputs);

    for(int i = 0; i < layer->num_outputs; i++) {
      layer->ordered_weights[i] = &layer->weights[i * layer->num_neurons];
    }

  } else {
    layer->weights = NULL;
    layer->num_outputs = 0;
    layer->ordered_weights = NULL;
  }
}

void free_layer(Layer *layer) {
  free(layer->neuron_values);
  free(layer->neuron_biases);
  if(layer->weights != NULL) {
    free(layer->weights);
  }
  if(layer->ordered_weights != NULL) {
    free(layer->ordered_weights);
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
  int rows = layer->num_outputs; // M | the number of rows is the number of outputs
  int columns = layer->num_neurons; // K

  double *output_matrix;
  output_matrix = malloc(sizeof(double) * layer->num_outputs);
  
  // quick matrix multiplication of the weights and values
  // cblas_dgemm(
  //   CblasRowMajor, CblasNoTrans, CblasNoTrans,
  //   rows, 1, columns, // M, N, K
  //   1.0,
  //   layer->weights, rows,
  //   // layer->output_layer->neuron_values, columns,
  //   layer->neuron_values, columns,
  //   0.0,
  //   output_matrix, rows
  // );

  cblas_dgemm(
    CblasRowMajor,
    CblasNoTrans,
    CblasNoTrans,
    rows, 1, columns,
    1.0,
    layer->weights, columns,
    layer->neuron_values, 1,
    0.0,
    output_matrix, 1
  );

  // then add the bias and update the output values
  for(int i = 0; i < layer->num_outputs; i++) {
    layer->output_layer->neuron_values[i] = 1 / (1 + exp(-(output_matrix[i] + layer->output_layer->neuron_biases[i])));
  }

  free(output_matrix);
}

void evaluate_network(Network *network) {
  for(int i = 0; i < network->num_layers - 1; i++) {
    feed_forward(&network->layers[i]);
  }
}


double get_cost_funtion(Network *network, double *expected) {

  double *outputs = network->outputs->neuron_values;
  double num_outputs = network->num_outputs;

  double cost = 0;

  for(int i = 0; i < num_outputs; i++) {
    double temp = outputs[i] - expected[i];
    cost += temp * temp;
  }

  return cost;
}

void back_propogate(Network *network, double *expected, CostMap *costmap) {

  costmap->num_iterations += 1;
  costmap->cumulative_cost += get_cost_funtion(network, expected);

  // CostMap *costmap;
  // costmap = malloc(sizeof(CostMap));
  // initialise_cost_map(costmap, network);
  
  // get all of the values for the costmap

  // doing the back propogation gives the gradients of all of the parameters of the neural network
  // start at the outputs
  
  // to find the gradient, we need to find the derivative of all parameters relative to the cost

  // start from the output layer
  // how much do we need to change the weights and biases of the previous layer to effect the output and therefore cost?

  // for the weights:

  // 3b1b backpropogation DL4 : https://www.youtube.com/watch?v=tIeHLnjs5U8&ab_channel=3Blue1Brown
  // 4:10

  // double neuron_value = network->outputs->neuron_values[0];
  // // effect of the cost with respect to the value of the output neurons:
  // double dcdal = 2 * (neuron_value - expected[0]);
  //
  // // effect of the value of the output neuron with respect to z
  // // derivative of "squishing" function 1/(1 + exp(-z)) = (exp(-x))/(1 + exp(-x))^2 = neuron_value (1 - neuron_value)
  // double dadz = neuron_value * (1 - neuron_value);
  //
  // // effect of the value of the z value with respect to the weight
  // double dzdw = network->layers[network->num_layers - 2].neuron_values[0];
  //
  // double dcdw0 = dcdal * dadz * dzdw;

  // so then to do this for all of the weights at a given neuron:
  // dcdal stays the same
  // dadz stays the same
  // dzdw changes

  // work backwards through the layers
  for(int i = network->num_layers - 1; i >= 1; i--) {

    Layer layer = network->layers[i];
    Layer *input_layer =  layer.input_layer;
    Layer *output_layer = layer.output_layer;

    // work through all of the neurons in the layer
    for(int j = 0; j < layer.num_neurons; j++) {

      // I believe that all of these values have to be saved for use in the next layers in the chain rule
      
      double neuron_value = layer.neuron_values[j];

      // effect of the value of the output neuron with respect to z
      double dadz = neuron_value * (1 - neuron_value);
      // store this value
      costmap->layers[i].dadz[j] = dadz;

      double dcdal;

      if (i == network->num_layers - 1) {
        // effect of the cost with respect to the value of the output neurons
        dcdal = 2 * (neuron_value - expected[0]); // base case for the outut layer

      } else { // this gets propogated through in other cases

        // find the effect of the current value on the next value i.e. weight and multiply by dcdal of next value
        // however this is done by the sum over the next layer as this neuron effects the cost through all of the output neurons
        dcdal = 0;
        for(int k = 0; k < output_layer->num_neurons; k++) {
          dcdal += layer.ordered_weights[k][j] * costmap->layers[i + 1].cost_derivative_of_values[k] * costmap->layers[i + 1].dadz[k];
        }
      }
      // store this value
      costmap->layers[i].cost_derivative_of_values[j] = dcdal;

      // multiply these together
      double temp = dcdal * dadz;

      // go through all of the weights for this neuron
      for(int k = 0; k < input_layer->num_neurons; k++) {
        // effect of the value of the z value with respect to the weight
        double dzdw = layer.input_layer->neuron_values[k];
        double dcdw = temp * dzdw;

        // then put this into the cost map in its relative position
        costmap->layers[i - 1].ordered_weights[j][k] += dcdw; 
      }

      // the bias for this neuron is just temp
      costmap->layers[i].biases[j] += temp;
    }

  }

}


int apply_cost_map(Network *network, CostMap *costmap, double learning_rate) {
  // here the costmap needs to be applied to the network
  if (learning_rate <= 0) {
    return -1;
  }

  // loop through all of the layers in the network
  for(int i = 0; i < network->num_layers; i++) {
    Layer network_layer = network->layers[i];
    CostMapLayer costmap_layer = costmap->layers[i];

    // the scalar mult should take into account the numer of iterations taken in back propogation to get the average
    double scalar = -learning_rate / costmap->num_iterations; 

    // perform fast matrix action, all this does is make w_i = w_i - learning_rate * weight_gradients_i
    cblas_daxpy(network_layer.num_outputs * network_layer.num_neurons, 
                -scalar,
                costmap_layer.weights, 1,
                network_layer.weights, 1 
    );

    // then do the same for the bias
    cblas_daxpy(network_layer.num_neurons,
                -scalar,
                costmap_layer.biases, 1,
                network_layer.neuron_biases, 1
    );

  }

  return 0;
}


void initialise_cost_map(CostMap *costmap, Network *network) {

  costmap->num_iterations = 0;
  costmap->cumulative_cost = 0;
  costmap->num_layers = network->num_layers;
  costmap->num_outputs = network->num_outputs;
  costmap->num_inputs = network->num_inputs;
  costmap->hidden_layer_length = network->hidden_layer_length;

  costmap->layers = malloc(sizeof(CostMapLayer) *costmap->num_layers);

  initialise_cost_map_layer(&costmap->layers[0], costmap->num_inputs, costmap->hidden_layer_length);

  for(int i = 1; i < costmap->num_layers - 2; i++) {
    initialise_cost_map_layer(&costmap->layers[i], costmap->hidden_layer_length, costmap->hidden_layer_length);
  }

  initialise_cost_map_layer(&costmap->layers[costmap->num_layers - 2], costmap->hidden_layer_length, costmap->num_outputs);
  initialise_cost_map_layer(&costmap->layers[costmap->num_layers - 1], costmap->num_outputs, 0);

}

void free_cost_map(CostMap *costmap) {
  for(int i = 0; i < costmap->num_layers; i++) {
    free_cost_map_layer(&costmap->layers[i]);
  }
  free(costmap->layers);
  free(costmap);
}


void initialise_cost_map_layer(CostMapLayer *layer, int num_neurons, int num_outputs) {

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
  layer->dadz = malloc(sizeof(double) * num_neurons);
}

void free_cost_map_layer(CostMapLayer *layer) {
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


void gradient_descent_train(Network *network, double **inputs, double **expected_outputs, int num_inputs) {

  // create the cost map by back propogating over all of the inputs and outputs
  // first create a cost map
  CostMap *costmap = malloc(sizeof(CostMap));
  initialise_cost_map(costmap, network);

  // loop over all of the inputs
  for(int i = 0; i < num_inputs; i++) {
    // run the network for the given input
    // put in all of the inputs
    memcpy(network->inputs->neuron_values, inputs[i], sizeof(double) * network->num_inputs);
    // then evaluate the network
    evaluate_network(network);

    // we can then back propogate to train
    back_propogate(network, expected_outputs[i], costmap);
  }

  printf("cost function average: %f\n", costmap->cumulative_cost / costmap->num_iterations);

  // after training over all of these inputs, the costmap can be applied to the network
  apply_cost_map(network, costmap, 0.5); // TODO make learning rate variable

  // remove the costmap now that we are done
  free_cost_map(costmap);
}


// -=-==-=-==-=-=-=-=-=-=--==--= DEBUGGING FUNCTIONS
void print_output_layer_values(Network *network) {
  printf("[");
  for(int i = 0; i < network->num_outputs - 1; i++) {
    printf(" %f, ", network->outputs->neuron_values[i]);
  }
  printf(" %f]\n", network->outputs->neuron_values[network->num_outputs - 1]);
}





