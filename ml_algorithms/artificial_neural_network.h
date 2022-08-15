#ifndef ARTIFICIAL_NEURAL_NETWORK_H
#define ARTIFICIAL_NEURAL_NETWORK_H

#include "globals.h"
#include "machine_learning_model.h"
#include <map>

/**
 * @brief List of activation functions
 */
enum class ActivationFunction {
    SIGMOID = 0, /** The sigmoid function y = 1/(1+exp(-x)) */
    RELU = 1, /** The relu function y = max(0, x) */
    SOFTMAX = 2 /** The softmax function y = exp(x)/sum(exp(0:n)) */
};

/*
 * Neuron class definition
 */
class Neuron {
public:
    Neuron(real_t bias, real_vector_t weights);

    real_t get_bias() const;

    const real_vector_t &get_weights() const;

    friend std::ostream &operator<<(std::ostream &os, const Neuron &neuron);

    real_t compute_weighted_sum(real_vector_t &features_vector);

private:
    real_t bias;
    real_vector_t weights;
};

/*
 * ArtificialNeuralNetwork class definition
 */
class ArtificialNeuralNetwork : public MachineLearningModel {
public:
    ArtificialNeuralNetwork();

    const std::map<std::size_t, std::vector<Neuron>> &get_layers() const;

    const std::map<std::size_t, ActivationFunction> &get_layers_activation_function() const;

    const std::vector<std::string> &get_classes() const;

    friend std::ostream &operator<<(std::ostream &os, const ArtificialNeuralNetwork &artificial_neural_network);

    void fill_from_csv(const std::filesystem::path &csv_file_path) override;

    std::string predict_class(const real_vector_t &features_vector) override;

private:
    std::map<std::size_t, std::vector<Neuron>> layers;
    std::map<std::size_t, ActivationFunction> layers_activation_function;
    std::vector<std::string> classes;

    void add_layer(std::size_t layer_id, const std::vector<Neuron> &neurons, ActivationFunction activation_function);

    void remove_layer(std::size_t layer_id);

    void clear();

    real_t compute_layer_activation_function(std::size_t layer_ind, real_vector_t &weighted_sums, std::size_t weighted_sum_ind);

};

#endif //ARTIFICIAL_NEURAL_NETWORK_H
