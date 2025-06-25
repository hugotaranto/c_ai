#include "Training.h"
#include "Network.h"
#include "GradientMap.h"
#include "UtilAi.h"

//------------------------------------------------------------------------------------------------------------------------
// Training Functions
//------------------------------------------------------------------------------------------------------------------------
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

void back_propogate(Network *network, double *expected, GradientMap *gradientmap) {

  // update the stored number of iterations and the cost
  gradientmap->num_iterations += 1;
  gradientmap->cumulative_cost += get_cost_funtion(network, expected);

  // work backwards through the layers:
  for(int i = network->num_layers - 1; i >= 0; i--) {

    Layer network_layer = network->layers[i];
    GradientMapLayer gradientmap_layer = gradientmap->layers[i];
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
          double forward_value_derivative = gradientmap->layers[i + 1].cost_derivative_of_values[k];
          double forward_value_z_derivative = output_layer->neuron_values[k] * (1 - output_layer->neuron_values[k]);
          double weight_between_neurons = network_layer.ordered_weights[k][j];
          cost_value_derivative += forward_value_derivative * forward_value_z_derivative * weight_between_neurons;
        }

        // the weight derivative can then be calculated
        // loop through all of the output neurons
        for(int k = 0; k < output_layer->num_neurons; k++) {
          double output_neuron_value = network_layer.output_layer->neuron_values[k];
          double sigmoid_output_neuron_derivative = output_neuron_value * (1 - output_neuron_value);
          double weight_derivative = neuron_value * sigmoid_output_neuron_derivative * gradientmap->layers[i + 1].cost_derivative_of_values[k];
          gradientmap_layer.ordered_weights[k][j] += weight_derivative;
        }
      }

      gradientmap_layer.cost_derivative_of_values[j] = cost_value_derivative;

      // the bias for the current neuron can be calculated
      double sigmoid_derivative = neuron_value * (1 - neuron_value);
      double bias_derivative = sigmoid_derivative * cost_value_derivative;
      gradientmap_layer.biases[j] += bias_derivative;

    }

  }

}

void gradient_descent_train(Network *network, double **inputs, double **expected_outputs, int num_inputs, double learning_rate) {

  // create the cost map by back propogating over all of the inputs and outputs
  // first create a cost map
  GradientMap *gradientmap = malloc(sizeof(GradientMap));
  initialise_gradient_map(gradientmap, network);

  // loop over all of the inputs
  for(int i = 0; i < num_inputs; i++) {
    // run the network for the given input
    // put in all of the inputs
    memcpy(network->inputs->neuron_values, inputs[i], sizeof(double) * network->num_inputs);
    // then evaluate the network
    evaluate_network(network);

    // we can then back propogate to train
    back_propogate(network, expected_outputs[i], gradientmap);
  }

  printf("cost function average: %f\n", gradientmap->cumulative_cost / gradientmap->num_iterations);

  // after training over all of these inputs, the costmap can be applied to the network
  apply_gradient_map(network, gradientmap, learning_rate); // TODO make learning rate variable

  // remove the costmap now that we are done
  free_gradient_map(gradientmap);
}

int stochastic_gradient_descent_train(Network *network, NetworkTrainingData *training_data, int batch_size, int epoch_count, double learning_rate) {

  if (batch_size > training_data->num_data_points) {
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
  int *indices = malloc(sizeof(int) * training_data->num_data_points);
  for(int i = 0; i < training_data->num_data_points; i++) {
    indices[i] = i;
  }

  int batches_per_epoch = training_data->num_data_points / batch_size;

  // create the gradientmap
  GradientMap *gradientmap = malloc(sizeof(GradientMap));
  initialise_gradient_map(gradientmap, network);

  // then loop through the number of epochs
  for(int i = 0; i < epoch_count; i++) {
    // shuffle the indices
    shuffle_indices(indices, training_data->num_data_points);

    // then for each epoch we need to run n batches where n = number of data points / batch size
    // this gives us a little under 1 epoch -- could use some optimising to get closer or equal to 1 epoch


    for(int k = 0; k < batches_per_epoch; k++) {

      // create a new cost map
      // GradientMap *costmap = malloc(sizeof(GradientMap));
      // initialise_cost_map(costmap, network);

      // reset the cost map
      reset_gradient_map(gradientmap);

      int starting_index = k * batch_size;
      for(int j = starting_index; j < (starting_index + batch_size); j++) {

        int index = indices[j];

        // insert the current test data point into the network inputs
        // memcpy(void *dst, const void *src, size_t n)
        memcpy(network->inputs->neuron_values, training_data->inputs[index], sizeof(double) * network->num_inputs);

        // evaluate the network
        evaluate_network(network);

        // then back propagate
        back_propogate(network, training_data->outputs[index], gradientmap);
      }

      // once the batch has been trained just apply the gradientmap
      apply_gradient_map(network, gradientmap, learning_rate);


      // free the costmap ready for another batch
      // free_cost_map(costmap);
    }

    printf("Cost Function Average After Epoch %d: %f\n", i + 1, gradientmap->cumulative_cost / gradientmap->num_iterations);
  }
  
  free_gradient_map(gradientmap);
  free(indices);
  return 0;
}


