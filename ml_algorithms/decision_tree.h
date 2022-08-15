#ifndef DECISION_TREE_H
#define DECISION_TREE_H

#include <filesystem>
#include <map>
#include <string>
#include "machine_learning_model.h"
#include "globals.h"

/*
 * TreeNode class definition
 */
class TreeNode {
public:
    TreeNode(std::string class_name, real_t threshold, int feature_id, int left_children_id, int right_children_id);

    const std::string &get_class_name() const;

    real_t get_threshold() const;

    int get_feature_id() const;

    int get_left_children_id() const;

    int get_right_children_id() const;

    friend std::ostream &operator<<(std::ostream &os, const TreeNode &tree_node);

    bool as_children() const;

private:
    std::string class_name;
    real_t threshold;
    std::size_t feature_id;
    int left_children_id;
    int right_children_id;

};

/*
 * DecisionTree class definition
 */
class DecisionTree : public MachineLearningModel {
    // id 0 = root
    // left/right id = -1 -> no sub node
public:
    DecisionTree();

    const std::map<std::size_t, TreeNode> &get_tree() const;

    std::size_t get_depth() const;

    std::size_t get_depth();

    std::size_t get_number_of_nodes() const;

    int get_max_feature_id() const;

    TreeNode get_node(std::size_t node_id) const;

    friend std::ostream &operator<<(std::ostream &os, const DecisionTree &decision_tree);

    void insert_node(std::size_t node_id, const TreeNode &node);

    void remove_node(std::size_t node_id);

    void clear();

    void fill_from_csv(const std::filesystem::path &csv_file_path) override;

    std::string predict_class(const real_vector_t &features_vector) override;

private:
    std::map<std::size_t, TreeNode> tree;
    std::size_t depth;
    std::size_t number_of_nodes;
    int max_feature_id;
    bool depth_up_to_date;

    std::size_t compute_depth() const;
};

#endif //DECISION_TREE_H
