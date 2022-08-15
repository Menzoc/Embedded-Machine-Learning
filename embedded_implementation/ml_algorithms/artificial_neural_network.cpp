#include "../helpers/print_helpers.h"
#include "../helpers/file_helpers.h"
#include "../helpers/log.h"
#include "artificial_neural_network.h"

#include <utility>
#include <execution>
#include <iterator>

/*
 * Neuron class definition
 */

/* PUBLIC DEFINITION */
Neuron::Neuron(real_t bias, real_vector_t weights) :
        bias(bias), weights(std::move(weights)) {}

real_t Neuron::get_bias() const {
    return bias;
}

const real_vector_t &Neuron::get_weights() const {
    return weights;
}

std::ostream &operator<<(std::ostream &os, const Neuron &neuron) {
    return os << "(bias: " << neuron.get_bias() << ", weights (" << neuron.get_weights().size() << "): " << neuron.get_weights() << ")";
}

real_t Neuron::compute_weighted_sum(real_vector_t &features_vector) {
    if (features_vector.size() != this->weights.size()) {
        LOG(LOG_ERROR) << "Error : trying to compute the activation of a neuron but the feature vector size (" << features_vector.size() << ") is different from the neuron weights size (" << this->weights.size() << ")";
        throw std::invalid_argument("Feature vector size differ from neuron weights size!");
    }

    real_t weighted_sum = 0.0;
    std::for_each(std::execution::seq, this->weights.cbegin(), this->weights.cend(), [&features_vector, &weighted_sum, i = 0](real_t x) mutable {
        weighted_sum += features_vector.at(i) * x;
        i++;
    });
    weighted_sum += this->bias;

    return weighted_sum;
}


/*
 * ArtificialNeuralNetwork class definition
 */

/* PUBLIC DEFINITION */
ArtificialNeuralNetwork::ArtificialNeuralNetwork() {
    this->clear();
}

const std::map<std::size_t, std::vector<Neuron>> &ArtificialNeuralNetwork::get_layers() const {
    return layers;
}

const std::map<std::size_t, ActivationFunction> &ArtificialNeuralNetwork::get_layers_activation_function() const {
    return layers_activation_function;
}


const std::vector<std::string> &ArtificialNeuralNetwork::get_classes() const {
    return classes;
}

std::ostream &operator<<(std::ostream &os, const ArtificialNeuralNetwork &artificial_neural_network) {
    for (const std::pair<std::size_t, std::vector<Neuron>> layer: artificial_neural_network.get_layers()) {
        os << "layer " << layer.first << " (" << layer.second.size() << ")";
        switch (artificial_neural_network.get_layers_activation_function().at(layer.first)) {
            case ActivationFunction::SIGMOID : {
                os << " activation function SIGMOID";
                break;
            }
            case ActivationFunction::RELU : {
                os << " activation function RELU";
                break;
            }
            case ActivationFunction::SOFTMAX : {
                os << " activation function SOFTMAX";
                break;
            }
            default: {
                os << " activation function UNKNOWN";
                break;
            }
        }
        os << ": \n";
        for (const Neuron &neuron: layer.second) {
            os << "\t - " << neuron << "\n";
        }
    }
    return os;
}

void ArtificialNeuralNetwork::fill_from_csv(const std::filesystem::path &csv_folder_path) {
    const char delimiter = ',';
    const std::string csv_extension = ".csv";


    auto csv_files = alpha_files_listing(csv_folder_path.string(), csv_extension);
    for (std::size_t layer_i = 0; layer_i < csv_files.size(); layer_i++) {
        LOG(LOG_DEBUG) << "reading: " << layer_i << ", " << csv_files.at(layer_i);
        std::string line = {};
        std::string data = {};

        std::ifstream input_file(csv_files.at(layer_i));
        if (!input_file.is_open()) {
            throw std::filesystem::filesystem_error("Can't open file!", std::make_error_code(std::errc::no_such_file_or_directory));
        }

        //TODO: check file extension and header
        std::vector<Neuron> layer = {};
        std::vector<std::string> pred_classes = {};
        ActivationFunction layer_activation_function;
        bool header_skipped = false;
        while (std::getline(input_file, line)) {
            if (!header_skipped) {
                header_skipped = true;
                continue;
            } else {
                std::stringstream ss(line);
                size_t last = 0;
                size_t next = 0;
                // Get bias from line
                next = line.find(delimiter, last);
                real_t bias = std::stod(line.substr(last, next - last));
                last = next + 1;
                // Get the weigths from line
                real_vector_t weigths;
                while ((next = line.find(delimiter, last)) != std::string::npos) {
                    weigths.push_back((real_t) std::stod(line.substr(last, next - last)));
                    last = next + 1;
                }
                // If this is the output layer, the last value is the class names
                if (layer_i + 1 == csv_files.size()) {
                    std::string class_name = line.substr(last);
                    // Remove double quote characters around the class name
                    class_name.erase(remove(class_name.begin(), class_name.end(), '"'), class_name.end());
                    pred_classes.push_back(class_name);
                } else {
                    weigths.push_back((real_t) std::stod(line.substr(last)));
                }

                Neuron neuron = {bias, weigths};
                layer.push_back(neuron);
            }
        }
        // Use softmax only for last layer, else use Relu
        if (layer_i + 1 == csv_files.size()) {
            layer_activation_function = ActivationFunction::SOFTMAX;
        } else {
            layer_activation_function = ActivationFunction::RELU;
        }
        this->add_layer(layer_i, layer, layer_activation_function);
        this->classes = pred_classes;
    }
}

std::string ArtificialNeuralNetwork::predict_class(const real_vector_t &features_vector) {
    real_vector_t last_activations = features_vector;
    real_vector_t next_activations = features_vector;
    std::for_each(std::execution::seq, this->layers.cbegin(), this->layers.cend(), [&last_activations, &next_activations, this](const std::pair<std::size_t, std::vector<Neuron>> &pair)mutable {
        last_activations.clear();
        std::move(next_activations.cbegin(), next_activations.cend(), std::back_inserter(last_activations));
        next_activations.clear();
        real_vector_t weighted_sum = {};
        for (Neuron n: pair.second) {
            weighted_sum.push_back(n.compute_weighted_sum(last_activations));
        }
        for (std::size_t neuron_i = 0; neuron_i < weighted_sum.size(); neuron_i++) {
            switch (this->layers_activation_function.at(pair.first)) {
                case ActivationFunction::SIGMOID :
                case ActivationFunction::RELU : {
                    next_activations.push_back(this->compute_layer_activation_function(pair.first, weighted_sum, neuron_i));
                    break;
                }
                case ActivationFunction::SOFTMAX : {
                    next_activations.push_back(this->compute_layer_activation_function(pair.first, weighted_sum, neuron_i));
                    break;
                }
                default: {
                    LOG(LOG_ERROR) << "Error : the activation function value usage is not defined in the project";
                    throw std::domain_error("Unsupported activation function!");
                    break;
                }
            }

        }
    });

    // Get the prediction result with the most choice
    auto pr = std::max_element(std::execution::seq, next_activations.begin(), next_activations.end());

    return this->classes.at(std::distance(next_activations.begin(), pr));
}

/* PRIVATE DEFINITION */
void ArtificialNeuralNetwork::add_layer(std::size_t layer_id, const std::vector<Neuron> &neurons, ActivationFunction activation_function) {
    this->layers.insert(std::make_pair(layer_id, neurons));
    this->layers_activation_function.insert(std::make_pair(layer_id, activation_function));
}

void ArtificialNeuralNetwork::remove_layer(std::size_t layer_id) {
    this->layers.erase(layer_id);
    this->layers_activation_function.erase(layer_id);
}

void ArtificialNeuralNetwork::clear() {
    this->layers.clear();
    this->layers_activation_function.clear();
}

real_t ArtificialNeuralNetwork::compute_layer_activation_function(std::size_t layer_ind, real_vector_t &weighted_sums, std::size_t weighted_sum_ind) {
    switch (this->layers_activation_function.at(layer_ind)) {
        case ActivationFunction::SIGMOID : {
            return 1.0 / (1.0 + std::exp(-weighted_sums.at(weighted_sum_ind)));
            break;
        }
        case ActivationFunction::RELU : {
            return std::max((real_t) 0, weighted_sums.at(weighted_sum_ind));
            break;
        }
        case ActivationFunction::SOFTMAX : {
            return std::exp(weighted_sums.at(weighted_sum_ind)) / std::transform_reduce(std::execution::seq, weighted_sums.cbegin(), weighted_sums.cend(), 0.0, std::plus<>(), [](real_t r)mutable {
                return std::exp(r);
            });
            break;
        }
        default: {
            LOG(LOG_ERROR) << "Error : the activation function value usage is not defined in the project";
            throw std::domain_error("Unsupported activation function!");
            break;
        }
    }
    return 0;
}
