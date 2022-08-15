#ifndef RANDOM_FOREST_H
#define RANDOM_FOREST_H

#include <vector>
#include "decision_tree.h"
#include "machine_learning_model.h"
#include "../helpers/log.h"
#include "../helpers/file_helpers.h"

/*
 * RandomForest class definition
 */
class RandomForest : public MachineLearningModel {
public:

    RandomForest();

    const std::vector<DecisionTree> &get_trees() const;

    size_t get_number_of_trees() const;

    friend std::ostream &operator<<(std::ostream &os, const RandomForest &random_forest);

    void push_tree(const DecisionTree& tree);

    void pop_tree();

    void clear();

    void fill_from_csv(const std::filesystem::path &csv_folder_path) override;

    std::string predict_class(const real_vector_t& features_vector) override;

private:
    std::vector<DecisionTree> trees;

};

#endif //RANDOM_FOREST_H
