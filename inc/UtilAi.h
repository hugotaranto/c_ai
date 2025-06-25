#ifndef AI_UTIL_C
#define AI_UTIL_C

#include "ProjectForwards.h"


enum file_path_type {
  LOAD,
  SAVE
};

//------------------------------------------------------------------------------------------------------------------------
// Global Util Functions
//------------------------------------------------------------------------------------------------------------------------
void swap(int *a, int *b);
void shuffle_indices(int *indices, int num_indices);

int get_network_filepath(DynamicString *string, file_path_type type);

#endif
