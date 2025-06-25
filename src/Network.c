#include "Network.h"

//------------------------------------------------------------------------------------------------------------------------
// NetWork Functions
//------------------------------------------------------------------------------------------------------------------------
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

void free_network(Network *network) {
  if(network->layers) {
    for(int i = 0; i < network->num_layers; i++) {
      free_layer(&network->layers[i]);
    }
    free(network->layers);
  }
  free(network);
}

void evaluate_network(Network *network) {
  for(int i = 0; i < network->num_layers - 1; i++) {
    feed_forward(&network->layers[i]);
  }
}

void save_network_to_file(Network *network) {

  DynamicString *file_path = dstring_initialise();
  if (get_network_filepath(file_path, SAVE)) {
    dstring_free(file_path);
    return;
  }
  FILE *file = fopen(file_path->data, "wb");

  if(!file) {
    perror("couldn't open file to save network to");
    dstring_free(file_path);
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
  printf("File saved successfully to: %s\n", file_path->data);
  dstring_free(file_path);

}

Network* load_network_from_file() {

  DynamicString *file_path = dstring_initialise();
  if(get_network_filepath(file_path, LOAD)) {
    dstring_free(file_path);
    return NULL;
  }

  FILE *file = fopen(file_path->data, "rb");

  if(!file) {
    perror("Couldn't Open File");
    goto cleanup;
  }

  Network *network = malloc(sizeof(Network));
  if (!network) {
    perror("Network Allocation Failed");
    goto cleanup;
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
    goto cleanup;
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
      goto cleanup;
    }
    if(num_outputs != layer->num_outputs) {
      fprintf(stderr, "Number of Outputs in Layer %d Does not match saved data\n", i);
      goto cleanup;
    }
    if(num_inputs != layer->num_inputs) {
      fprintf(stderr, "Number of Inputs in Layer %d Does not match saved data\n", i);
      goto cleanup;
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
  printf("Successfully read Network From %s\n", file_path->data);
  dstring_free(file_path);
  return network;


cleanup:
  if(file) {
    fclose(file);
  }
  if(file_path) {
    dstring_free(file_path);
  }
  if(network) {
    free_network(network);
  }
  return NULL;
}


//------------------------------------------------------------------------------------------------------------------------
// Layer Functions
//------------------------------------------------------------------------------------------------------------------------
void initialise_layer(Layer *layer, int num_neurons, Layer *input_layer, Layer *output_layer) {

  layer->num_neurons = num_neurons;
  layer->neuron_values = malloc(sizeof(double) * layer->num_neurons);
  layer->neuron_biases = malloc(sizeof(double) * layer->num_neurons);
  layer->output_layer = output_layer;
  layer->input_layer = input_layer;

  if (input_layer == NULL) {
    layer->num_inputs = 0;
  }

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

      // randomise the weight values
      for(int j = 0; j < num_neurons; j++) {
        double limit = 1.0 / sqrt((double) num_neurons);
        layer->ordered_weights[i][j] = (drand48() * 2 * limit) - limit;
      }
    }

    // set the number of inputs for the next layer
    output_layer->num_inputs = num_neurons;

  } else {
    layer->weights = NULL;
    layer->num_outputs = 0;
    layer->ordered_weights = NULL;
  }
}

void free_layer(Layer *layer) {
  if(layer->neuron_values) {
    free(layer->neuron_values);
  }
  if(layer->neuron_biases) {
  free(layer->neuron_biases);
  }
  if(layer->weights != NULL) {
    free(layer->weights);
  }
  if(layer->ordered_weights != NULL) {
    free(layer->ordered_weights);
  }
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

//------------------------------------------------------------------------------------------------------------------------
// Debugging Functions
//------------------------------------------------------------------------------------------------------------------------
void print_output_layer_values(Network *network) {
  printf("[");
  for(int i = 0; i < network->num_outputs - 1; i++) {
    printf(" %f, ", network->outputs->neuron_values[i]);
  }
  printf(" %f]\n", network->outputs->neuron_values[network->num_outputs - 1]);
}


