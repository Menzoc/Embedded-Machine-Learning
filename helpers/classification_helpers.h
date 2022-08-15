#ifndef CLASSIFICATION_HELPERS_H
#define CLASSIFICATION_HELPERS_H

#include <string>
#include <vector>
#include <map>
#include <execution>
#include "../ml_algorithms/machine_learning_model.h"
#include "../helpers/log.h"
#include "../helpers/print_helpers.h"

// <true_label, predicted_label>
static inline real_t predictions_report(const std::vector<std::pair<std::string, std::string>> &predictions) {
    std::size_t good_predictions = 0;
    std::for_each(std::execution::seq, predictions.cbegin(), predictions.cend(), [&good_predictions](const std::pair<std::string, std::string> &pair)mutable {
        if (pair.first == pair.second) {
            good_predictions += 1;
        }
    });

    return (real_t) good_predictions / (real_t) predictions.size();
}

// <true_label, predicted_label>
static inline void show_confusion_matrix(const std::vector<std::pair<std::string, std::string>> &predictions) {
    std::map<std::string, std::map<std::string, int>> confusion_matrix;
    std::size_t longest_unique_label_size = 0;
    std::set<std::string> unique_labels;

    // Find unique labels
    std::for_each(std::execution::seq, predictions.cbegin(), predictions.cend(), [&unique_labels, &longest_unique_label_size](const std::pair<std::string, std::string> &pair) mutable {
        if (!unique_labels.contains(pair.first)) {
            unique_labels.insert(pair.first);
            longest_unique_label_size = pair.first.size() > longest_unique_label_size ? pair.first.size() : longest_unique_label_size;
        }
        if (!unique_labels.contains(pair.second)) {
            unique_labels.insert(pair.second);
            longest_unique_label_size = pair.second.size() > longest_unique_label_size ? pair.second.size() : longest_unique_label_size;
        }
    });

    // Generate empty 2D hashmap
    std::for_each(std::execution::seq, unique_labels.cbegin(), unique_labels.cend(), [&confusion_matrix, &unique_labels](const std::string &label) mutable {
        confusion_matrix.insert(std::make_pair(label, std::map<std::string, int>{}));

        std::transform(unique_labels.cbegin(), unique_labels.cend(),
                       std::inserter(confusion_matrix.at(label), confusion_matrix.at(label).end()),
                       [](const std::string &unique_label) { return std::make_pair(unique_label, 0); }
        );
    });

    // Fill the 2D hashmap
    std::for_each(std::execution::seq, predictions.cbegin(), predictions.cend(), [&confusion_matrix](const std::pair<std::string, std::string> &pair) mutable {
        confusion_matrix.at(pair.first).at(pair.second) += 1;
    });

    std::cout << (std::string(longest_unique_label_size + 5, ' ') + "\t ");
    for (std::string label: unique_labels) {
        try {
            std::cout << (char) toupper(label.at(0));
            std::cout << label.at(1);
        } catch (std::out_of_range &e) {
        }
        std::cout << "\t";
    }
    std::cout << std::endl;
    for (const auto &confusion_matrix_line: confusion_matrix) {
        int space_len = (int) longest_unique_label_size - (int) confusion_matrix_line.first.size();
        try {
            std::cout << "(" << (char) toupper(confusion_matrix_line.first.at(0));
            std::cout << confusion_matrix_line.first.at(1);
        } catch (std::out_of_range &e) {
        }
        std::cout << ") " << confusion_matrix_line.first << std::string(space_len, ' ') << "\t[";
        std::size_t i = 0;
        for (const auto &confusion_matrix_line_i: confusion_matrix_line.second) {
            if (i >= confusion_matrix_line.second.size() - 1) {
                std::cout << (std::to_string(confusion_matrix_line_i.second));
            } else {
                std::cout << (std::to_string(confusion_matrix_line_i.second) + "\t");
            }
            i++;
        }
        std::cout << "]" << std::endl;
    }

}

// <true_label, predicted_label>
static inline std::vector<std::pair<std::string, std::string>> make_predictions(MachineLearningModel &model, const std::vector<std::pair<std::string, real_vector_t>> &feature_vectors) {
    std::vector<std::pair<std::string, std::string>> predictions;

    // Predict for all feature vectors
    auto start_time = std::chrono::high_resolution_clock::now();
    LOG(LOG_INFO) << "Making " << feature_vectors.size() << " prediction using the machine learning model...";
    std::for_each(std::execution::seq, feature_vectors.cbegin(), feature_vectors.cend(), [&predictions, &model](const std::pair<std::string, real_vector_t> &pair)mutable {
        std::string predicted_class = model.predict_class(pair.second);
        LOG(LOG_DEBUG) << "\ttrue class: " << pair.first << ", predicted class: " << predicted_class;
        predictions.emplace_back(pair.first, predicted_class);
    });
    auto stop_time = std::chrono::high_resolution_clock::now();
    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(stop_time - start_time).count();
    LOG(LOG_INFO) << "Prediction done in " << elapsed_time / 1000 << "s and " << elapsed_time % 1000 << "ms";

    return predictions;
}

#endif //CLASSIFICATION_HELPERS_H
