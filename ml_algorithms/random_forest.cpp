
#include <vector>
#include <map>
#include <execution>
#include "random_forest.h"


/*
 * RandomForest class definition
 */

/* PUBLIC DEFINITION */
RandomForest::RandomForest() {
    trees = {};
}

const std::vector<DecisionTree> &RandomForest::get_trees() const {
    return trees;
}

unsigned long RandomForest::get_number_of_trees() const {
    return this->trees.size();
}

std::ostream &operator<<(std::ostream &os, const RandomForest &random_forest) {
    for (const DecisionTree &tree: random_forest.get_trees()) {
        os << "\n - tree[depth: " << tree.get_depth() << ", number_of_nodes: " << tree.get_number_of_nodes() << "]";
    }
    return os;
}

void RandomForest::push_tree(const DecisionTree &tree) {
    this->trees.push_back(tree);
}

void RandomForest::pop_tree() {
    this->trees.pop_back();
}

void RandomForest::clear() {
    this->trees.clear();
}

void RandomForest::fill_from_csv(const std::filesystem::path &csv_folder_path) {
    auto csv_files = alpha_files_listing(csv_folder_path.string());
    for (const auto &csv_file_path: csv_files) {
        DecisionTree new_tree = {};
        new_tree.fill_from_csv(csv_file_path);
        this->push_tree(new_tree);
    }
}

std::string RandomForest::predict_class(const real_vector_t &features_vector) {
    std::map<std::string, std::size_t> pred_results;
    using pred_results_pair_t = decltype(pred_results)::value_type;

    // Get results for each tree
    std::for_each(std::execution::seq, this->trees.begin(), this->trees.end(), [&pred_results, &features_vector](DecisionTree &tree) mutable {
        std::string pred_result = tree.predict_class(features_vector);
        if (pred_results.count(pred_result) == 0) {
            pred_results.insert(std::make_pair(pred_result, 1));
        } else {
            pred_results.at(pred_result) += 1;
        }
    });

    // Get the prediction result with the most choice
    auto pr = std::max_element(std::execution::seq, pred_results.cbegin(), pred_results.cend(),
                               [](const pred_results_pair_t &p1, const pred_results_pair_t &p2) {
                                   return p1.second < p2.second;
                               });

    return pr->first;
}

/* PRIVATE DEFINITION */
