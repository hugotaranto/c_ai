#include "UtilAi.h"

//------------------------------------------------------------------------------------------------------------------------
// Global Util Functions
//------------------------------------------------------------------------------------------------------------------------
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

int get_network_filepath(DynamicString *file_path, file_path_type type) {
  while(1) {
    if(type == LOAD) {
      printf("Please Enter the path to a file to load (e.g. ../networks/xxx) or 'exit' to quit: ");
    } else {
      printf("Please Enter the path to a file to save to (e.g. ../networks/xxx) or 'exit' to quit: ");
    }
    
    // get the input
    dstring_readline(file_path, stdin);
    // convert it to lower
    DynamicString *lower = dstring_to_lower(file_path);

    // check if it is exit:
    if(strncmp(lower->data, "exit", lower->length) == 0) {
      dstring_free(lower);
      return -1;
    }
    dstring_free(lower);

    if(dstring_contains_file_extension(file_path)) {
      printf("Please Do Not Include the extension (i.e. '.nn') This will be added automatically\n");
      continue;
    }

    // add the extension on
    dstring_combine_simple(file_path, ".nn");

    if(type == LOAD) {
      // check if the file exists:
      FILE *file = fopen(file_path->data, "rb");
      if(!file) {
        printf("File: '%s' Could not be opened\n", file_path->data);
        continue;
      } else {
        fclose(file);
        return 0;
      }
    } else {
      // need to check that the path can be saved to
      FILE *file = fopen(file_path->data, "wb");
      if(!file) {
        printf("The file could not be saved to %s\n", file_path->data);
        continue;
      } else {
        fclose(file);
        return 0;
      }
    }

  }

  return 0;
}

