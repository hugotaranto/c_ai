#include "ai.h"
#include <cblas.h>
#include <time.h>
#include <math.h>

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

  // randomise the biases
  for(int i = 0; i < num_neurons; i++) {
    layer->neuron_biases[i] = drand48() -0.5; // between [-0.5, 0.5]
  }

  if (output_layer != NULL) {
    layer->num_outputs = output_layer->num_neurons;
    layer->weights = malloc(sizeof(double) * layer->num_outputs * layer->num_neurons);
    layer->ordered_weights = malloc(sizeof(double*) * layer->num_outputs);

    for(int i = 0; i < layer->num_outputs; i++) {
      layer->ordered_weights[i] = &layer->weights[i * layer->num_neurons];
      for(int j = 0; j < num_neurons; j++) {
        double limit = 1.0 / sqrt((double) num_neurons);
        layer->ordered_weights[i][j] = (drand48() * 2 * limit) - limit;
      }
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

  // update the stored number of iterations and the cost
  costmap->num_iterations += 1;
  costmap->cumulative_cost += get_cost_funtion(network, expected);

  // work backwards through the layers:
  for(int i = network->num_layers - 1; i >= 0; i--) {

    Layer network_layer = network->layers[i];
    CostMapLayer costmap_layer = costmap->layers[i];
    Layer *output_layer = network_layer.output_layer;

    // loop through all of the neurons in that layer
    for(int j = 0; j < network_layer.num_neurons; j++) {

      double neuron_value = network_layer.neuron_values[j];

      double cost_value_derivative = 0;

      // base case for output layer:
      if(i == network->num_layers - 1) {

        // the derivative of this value to the cost is the derivative of the cost function at each point
        // i.e. the cost function is sum(y - y_expected)^2 -- so the derivative is 2 (y - y_expected)
        cost_value_derivative = 2 * (neuron_value - expected[j]);
        // then these just need to be stored for back propagating to the hidden layers:

      } else {


        // need to loop through all of the L + 1 (next layer neurrons)
        for(int k = 0; k < output_layer->num_neurons; k++) {
          // this is L + 1 (L)
          double forward_value_derivative = costmap->layers[i + 1].cost_derivative_of_values[k];
          double forward_value_z_derivative = output_layer->neuron_values[k] * (1 - output_layer->neuron_values[k]);
          double weight_between_neurons = network_layer.ordered_weights[k][j];
          cost_value_derivative += forward_value_derivative * forward_value_z_derivative * weight_between_neurons;
        }

        // the weight derivative can then be calculated
        // loop through all of the output neurons
        for(int k = 0; k < output_layer->num_neurons; k++) {
          double output_neuron_value = network_layer.output_layer->neuron_values[k];
          double sigmoid_output_neuron_derivative = output_neuron_value * (1 - output_neuron_value);
          double weight_derivative = neuron_value * sigmoid_output_neuron_derivative * costmap->layers[i + 1].cost_derivative_of_values[k];
          costmap_layer.ordered_weights[k][j] += weight_derivative;
        }
      }

      costmap_layer.cost_derivative_of_values[j] = cost_value_derivative;

      // the bias for the current neuron can be calculated
      double sigmoid_derivative = neuron_value * (1 - neuron_value);
      double bias_derivative = sigmoid_derivative * cost_value_derivative;
      costmap_layer.biases[j] += bias_derivative;

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
    Layer *network_layer = &network->layers[i];
    CostMapLayer *costmap_layer = &costmap->layers[i];

    // the scalar mult should take into account the numer of iterations taken in back propogation to get the average
    double scalar = -learning_rate / costmap->num_iterations; 

    // perform fast matrix action, all this does is make w_i = w_i - learning_rate * weight_gradients_i
    cblas_daxpy(network_layer->num_outputs * network_layer->num_neurons, 
                scalar,
                costmap_layer->weights, 1,
                network_layer->weights, 1 
    );

    // then do the same for the bias
    cblas_daxpy(network_layer->num_neurons,
                scalar,
                costmap_layer->biases, 1,
                network_layer->neuron_biases, 1
    );

  }

  return 0;
}


void gradient_descent_train(Network *network, double **inputs, double **expected_outputs, int num_inputs, double learning_rate) {

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
  apply_cost_map(network, costmap, learning_rate); // TODO make learning rate variable

  // remove the costmap now that we are done
  free_cost_map(costmap);
}


int stochastic_gradient_descent_train(Network *network, NetworkTestData *test_data, int batch_size, int epoch_count, double learning_rate) {

  if (batch_size > test_data->num_data_points) {
    perror("Batch Size Bigger Than Data Set");
    return -1;
  }
  if (batch_size <= 0) {
    perror("Batch Size Cannot Be less than 1");
    return -1;
  }

  // need to randomly select batch_size amount of test data points
  // then gradient descent train on that
  // then repeat for the number of epochs
  
  // make the indice array
  int *indices = malloc(sizeof(int) * test_data->num_data_points);
  for(int i = 0; i < test_data->num_data_points; i++) {
    indices[i] = i;
  }

  int batches_per_epoch = test_data->num_data_points / batch_size;

  // create the costmap
  CostMap *costmap = malloc(sizeof(CostMap));
  initialise_cost_map(costmap, network);

  // then loop through the number of epochs
  for(int i = 0; i < epoch_count; i++) {
    // shuffle the indices
    shuffle_indices(indices, test_data->num_data_points);

    // then for each epoch we need to run n batches where n = number of data points / batch size
    // this gives us a little under 1 epoch -- could use some optimising to get closer or equal to 1 epoch


    for(int k = 0; k < batches_per_epoch; k++) {

      // create a new cost map
      // CostMap *costmap = malloc(sizeof(CostMap));
      // initialise_cost_map(costmap, network);

      // reset the cost map
      reset_cost_map(costmap);

      int starting_index = k * batch_size;
      for(int j = starting_index; j < (starting_index + batch_size); j++) {

        int index = indices[j];

        // insert the current test data point into the network inputs
        // memcpy(void *dst, const void *src, size_t n)
        memcpy(network->inputs->neuron_values, test_data->inputs[index], sizeof(double) * network->num_inputs);

        // evaluate the network
        evaluate_network(network);

        // then back propagate
        back_propogate(network, test_data->outputs[index], costmap);
      }

      // once the batch has been trained just apply the costmap
      apply_cost_map(network, costmap, learning_rate);


      // free the costmap ready for another batch
      // free_cost_map(costmap);
    }

    printf("Cost Function Average After Epoch %d: %f\n", i + 1, costmap->cumulative_cost / costmap->num_iterations);
  }
  
  free_cost_map(costmap);
  free(indices);
  return 0;
}

void swap(int *a, int *b) {
  int temp = *a;
  *a = *b;
  *b = temp;
}

void shuffle_indices(int *indices, int num_indices) {
  for(int i = 0; i < num_indices; i++) {
    int j = rand() % (i + 1);
    swap(&indices[i], &indices[j]);
  }
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

void reset_cost_map(CostMap *costmap) {
  costmap->num_iterations = 0;
  costmap->cumulative_cost = 0;

  for(int i = 0; i < costmap->num_layers; i++) {
    CostMapLayer *layer = &costmap->layers[i];
    memset(layer->weights, 0, sizeof(double) * layer->num_outputs * layer->num_neurons);
    memset(layer->biases, 0, sizeof(double) * layer->num_neurons);
    memset(layer->cost_derivative_of_values, 0, sizeof(double) * layer->num_neurons);
  }
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
  memset(layer->cost_derivative_of_values, 0, sizeof(double) * num_neurons);
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

void save_network_to_file(Network *network) {

  char filename[256];
  printf("Enter path and filename to save. (e.g. ../network): ");
  scanf("%255s", filename);  // avoid buffer overflow

  // Combine with current directory (optional)
  char fullpath[512];
  snprintf(fullpath, sizeof(fullpath), "./%s.nn", filename); // saves in current folder

  FILE *file = fopen(fullpath, "wb");
  if (file == NULL) {
    perror("Failed to open file");
    return;
  }

  // save metadata
  fwrite(&network->num_layers, sizeof(int), 1, file);
  fwrite(&network->num_hidden_layers, sizeof(int), 1, file);
  fwrite(&network->hidden_layer_length, sizeof(int), 1, file);
  fwrite(&network->num_inputs, sizeof(int), 1, file);
  fwrite(&network->num_outputs, sizeof(int), 1, file);

  // save each of the layers
  for(int i = 0; i < network->num_layers; i++) {

    Layer *layer = &network->layers[i];

    // write the metadata for the layer
    fwrite(&layer->num_neurons, sizeof(int), 1, file);
    fwrite(&layer->num_outputs, sizeof(int), 1, file);
    fwrite(&layer->num_inputs, sizeof(int), 1, file);

    // write the biases
    fwrite(layer->neuron_biases, sizeof(double), layer->num_neurons, file);

    // write the weights
    if(layer->num_outputs > 0) {
      fwrite(layer->weights, sizeof(double), layer->num_neurons * layer->num_outputs, file);
    }

  }

  fclose(file);
  printf("File saved successfully to: %s\n", fullpath);
}

Network* load_network_from_file() {

  char filename[256];
  printf("Enter path and filename to load (DO NOT INCLUDE '.nn'). (e.g. ../network): ");
  scanf("%255s", filename);  // avoid buffer overflow

  // Combine with current directory (optional)
  char fullpath[512];
  snprintf(fullpath, sizeof(fullpath), "./%s.nn", filename); // saves in current folder

  FILE *file = fopen(fullpath, "rb");
  if (!file) {
    perror("Failed to open file");
    return NULL;
  }

  Network *network = malloc(sizeof(Network));
  if (!network) {
    perror("Network Allocation Failed");
    fclose(file);
    return NULL;
  }

  int num_layers, num_hidden_layers, hidden_layer_length, num_inputs, num_outputs;

  // read the network metadata
  fread(&num_layers, sizeof(int), 1, file);
  fread(&num_hidden_layers, sizeof(int), 1, file);
  fread(&hidden_layer_length, sizeof(int), 1, file);
  fread(&num_inputs, sizeof(int), 1, file);
  fread(&num_outputs, sizeof(int), 1, file);

  if (initialise_network(network, num_inputs, num_outputs, num_hidden_layers, hidden_layer_length)) {
    perror("Network Initilisation Failed");
    fclose(file);
    return NULL;
  }

  // now we just need to read in the layer data:
  for (int i = 0; i < network->num_layers; i++) {
    Layer *layer = &network->layers[i];

    int num_neurons, num_outputs, num_inputs;
    fread(&num_neurons, sizeof(int), 1, file);
    fread(&num_outputs, sizeof(int), 1, file);
    fread(&num_inputs, sizeof(int), 1, file);

    if(num_neurons != layer->num_neurons) {
      fprintf(stderr, "Number of Neurons in Layer %d Does not match saved data\n", i);
      fclose(file);
      return NULL;
    }
    if(num_outputs != layer->num_outputs) {
      fprintf(stderr, "Number of Outputs in Layer %d Does not match saved data\n", i);
      fclose(file);
      return NULL;
    }
    if(num_inputs != layer->num_inputs) {
      fprintf(stderr, "Number of Inputs in Layer %d Does not match saved data\n", i);
      fclose(file);
      return NULL;
    }

    // load the biases
    fread(layer->neuron_biases, sizeof(double), layer->num_neurons, file);

    // load the weights if this is not an ouput layer
    if (layer->num_outputs > 0) {
      int num_weights = layer->num_outputs * layer->num_neurons;
      fread(layer->weights, sizeof(double), num_weights, file);
    }
  }

  fclose(file);
  printf("Successfully read Network From %s\n", fullpath);
  return network;
}

// -=-==-=-==-=-=-=-=-=-=--==--= DEBUGGING FUNCTIONS
void print_output_layer_values(Network *network) {
  printf("[");
  for(int i = 0; i < network->num_outputs - 1; i++) {
    printf(" %f, ", network->outputs->neuron_values[i]);
  }
  printf(" %f]\n", network->outputs->neuron_values[network->num_outputs - 1]);
}


// Unfinished functions for testing - (fixed back prop before I had to test)
void make_back_prop_testcase() {


  Network *test_network = malloc(sizeof(Network));
  initialise_network(test_network, 1, 1, 1, 1);


  printf("setting up the inputs and outputs\n");
  // set up the inputs and outputs
  double **input = malloc(sizeof(double*));
  double **expected_output = malloc(sizeof(double*));

  input[0] = malloc(sizeof(double));
  expected_output[0] = malloc(sizeof(double));

  input[0][0] = 1;
  expected_output[0][0] = 0;

  printf("about to set weights and biases\n");

  // initialise the weights and biases of the network
  test_network->layers[0].ordered_weights[0][0] = 0.5;
  test_network->layers[1].ordered_weights[0][0] = 0.2;

  test_network->layers[1].neuron_biases[0] = 0.3;
  test_network->layers[2].neuron_biases[0] = 0.7;

  printf("evaluating\n");

  // set the input value
  test_network->inputs->neuron_values[0] = 1;
  // the evaluation can then be run
  evaluate_network(test_network);

  printf("Output Value: ");
  print_output_layer_values(test_network);

  //print the error
  printf("Error/Loss thingo with expected value 0: %f\n", get_cost_funtion(test_network, expected_output[0]));

  // now the tricky bit, testing the gradient descent
  CostMap *costmap = malloc(sizeof(CostMap));
  initialise_cost_map(costmap, test_network);

  back_propogate(test_network, expected_output[0], costmap);

  // print out the costmap

  free(input[0]);
  free(expected_output[0]);
  free(input);
  free(expected_output);

  free_network(test_network); 

}


void print_costmap_gradients(CostMap *costmap) {

  printf("Input Layer:\n");
  printf("Weights:\n");
  for(int i = 0; i < costmap->num_inputs; i++) {
    for(int k = 0; k < costmap->layers[1].num_neurons; k++) {
      printf("ordered_weights[%d][%d] = %f\n", k, i, costmap->layers[0].ordered_weights[k][i]);
    }
  }

  for(int layer = 1; layer < costmap->num_layers - 2; layer++) {

    printf("Hidden Layer %d\n", layer);
    printf("Biases:\n");

  }

}

