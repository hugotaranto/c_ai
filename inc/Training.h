#ifndef C_AI_TRAINING_H
#define C_AI_TRAINING_H

#include "ProjectForwards.h"

typedef struct NetworkTrainingData NetworkTrainingData;

struct NetworkTrainingData {
  int num_data_points;
  int input_length;
  double **inputs;

  int output_length;
  double **outputs;
};

//------------------------------------------------------------------------------------------------------------------------
// Training Functions
//------------------------------------------------------------------------------------------------------------------------
double get_cost_funtion(Network *network, double *expected);
void back_propogate(Network *network, double *expected, GradientMap *gradientmap);
void gradient_descent_train(Network *network, double **inputs, double **expected_outputs, int num_inputs, double learning_rate);
int stochastic_gradient_descent_train(Network *network, NetworkTrainingData *training_data, int batch_size, int epoch_count, double learning_rate);


#endif
