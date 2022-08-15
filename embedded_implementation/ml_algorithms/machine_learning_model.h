#ifndef MACHINE_LEARNING_MODEL_H
#define MACHINE_LEARNING_MODEL_H

#include <string>
#include <filesystem>
#include "globals.h"

class MachineLearningModel {
public:
    virtual void fill_from_csv(const std::filesystem::path &csv_folder_path) = 0;

    virtual std::string predict_class(const real_vector_t &features_vector) = 0;
};

#endif //MACHINE_LEARNING_MODEL_H
