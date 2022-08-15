#include "decision_tree.h"

#include <utility>
#include <functional>
#include <fstream>
#include "../helpers/log.h"

/*
 * TreeNode class definition
 */

/* PUBLIC DEFINITION */
TreeNode::TreeNode(std::string class_name, real_t threshold, int feature_id, int left_children_id, int right_children_id) :
        class_name(std::move(class_name)), threshold(threshold), feature_id(feature_id), left_children_id(left_children_id), right_children_id(right_children_id) {
}

const std::string &TreeNode::get_class_name() const {
    return class_name;
}

real_t TreeNode::get_threshold() const {
    return threshold;
}

int TreeNode::get_feature_id() const {
    return feature_id;
}

int TreeNode::get_left_children_id() const {
    return left_children_id;
}

int TreeNode::get_right_children_id() const {
    return right_children_id;
}

std::ostream &operator<<(std::ostream &os, const TreeNode &tree_node) {
    return os << "(class: " << tree_node.get_class_name() << ", treshold: " << tree_node.get_threshold() << ", feature_id: " << tree_node.get_feature_id() << ", left_id: " << tree_node.get_left_children_id() << ", right_id: " << tree_node.get_right_children_id() << ")";
}

bool TreeNode::as_children() const {
    if (this->left_children_id != -1 || this->right_children_id != -1) {
        return true;
    } else {
        return false;
    }
}


/*
 * DecisionTree class definition
 */

/* PUBLIC DEFINITION */
DecisionTree::DecisionTree() {
    this->tree.clear();
    this->depth = 0;
    this->number_of_nodes = 0;
    this->depth_up_to_date = true;
    this->max_feature_id = 0;
}

const std::map<std::size_t, TreeNode> &DecisionTree::get_tree() const {
    return this->tree;
}

size_t DecisionTree::get_depth() {
    if (this->depth_up_to_date) {
        return this->depth;
    } else {
        this->depth = this->compute_depth();
        this->depth_up_to_date = true;
        return this->depth;
    }
}

size_t DecisionTree::get_depth() const {
    if (this->depth_up_to_date) {
        return this->depth;
    } else {
        return compute_depth();
    }
}

std::size_t DecisionTree::get_number_of_nodes() const {
    return number_of_nodes;
}

int DecisionTree::get_max_feature_id() const {
    return this->max_feature_id;
}

TreeNode DecisionTree::get_node(std::size_t node_id) const {
    return this->tree.at(node_id);
}

std::ostream &operator<<(std::ostream &os, const DecisionTree &decision_tree) {
    for (const std::pair<const unsigned long, TreeNode> &tree: decision_tree.get_tree()) {
        os << "\n- node[" << tree.first << "]: " << tree.second;
    }
    return os;
}

void DecisionTree::insert_node(std::size_t node_id, const TreeNode &node) {
    if (this->tree.count(node_id) == 1) {
        if (node_id == 0) {
            LOG(LOG_ERROR) << "Error : trying to insert a node with id=0 but the tree already as a root node";
            throw std::invalid_argument("Tree already have a root node!");
        } else {
            LOG(LOG_ERROR) << "Error : trying to insert a node with id=" + std::to_string(node_id) + " but the tree already as a node with this id";
            throw std::invalid_argument("Tree already have a node with id " + std::to_string(node_id) + "!");
        }
    }
    this->tree.insert(std::make_pair(node_id, node));
    this->max_feature_id = std::max(node.get_feature_id(), this->max_feature_id);

    this->depth_up_to_date = false;
    this->number_of_nodes += 1;
}

void DecisionTree::remove_node(std::size_t node_id) {
    if (this->tree.count(node_id) == 0) {
        LOG(LOG_ERROR) << "Error : trying to remove a node with id=" + std::to_string(node_id) + " but the tree does not have a node with this id";
        throw std::invalid_argument("Tree does not have a node with id " + std::to_string(node_id) + "!");
    }
    if (this->tree.at(node_id).as_children()) {
        LOG(LOG_ERROR) << "Error : trying to remove the node with id=" + std::to_string(node_id) + " but this node as at least one children, remove children nodes first (left id=" + std::to_string(this->tree.at(node_id).get_left_children_id()) + ", right_id=" + std::to_string(this->tree.at(node_id).get_right_children_id()) + " !";
        throw std::invalid_argument("Node as children!");

    }
    this->tree.erase(node_id);

    this->depth_up_to_date = false;
    this->number_of_nodes -= 1;
}

void DecisionTree::clear() {
    this->tree.clear();

    this->depth = 0;
    this->depth_up_to_date = true;
    this->number_of_nodes = 0;
    this->max_feature_id = 0;
}

void DecisionTree::fill_from_csv(const std::filesystem::path &csv_file_path) {
    const char delimiter = ',';
    std::string line = {};
    std::string data = {};

    std::ifstream input_file(csv_file_path);
    if (!input_file.is_open()) {
        LOG(LOG_ERROR) << "Error : file with path " + csv_file_path.string() + "  not found.";
        throw std::filesystem::filesystem_error("Can't open file!", std::make_error_code(std::errc::no_such_file_or_directory));
    }

    //TODO: check file extension and header

    bool header_skipped = false;
    while (std::getline(input_file, line)) {
        if (!header_skipped) {
            header_skipped = true;
            continue;
        } else {
            std::stringstream ss(line);
            size_t last = 0;
            size_t next = 0;
            // Get tree node id from line
            next = line.find(delimiter, last);
            int node_id = std::stoi(line.substr(last, next - last));
            last = next + 1;
            // Get threshold from line
            next = line.find(delimiter, last);
            real_t threshold = (real_t) std::stod(line.substr(last, next - last));
            last = next + 1;
            // Get the feature id from line
            next = line.find(delimiter, last);
            int feature_id = std::stoi(line.substr(last, next - last));
            last = next + 1;
            // Get left children id from line
            next = line.find(delimiter, last);
            int left_children_id = std::stoi(line.substr(last, next - last));
            last = next + 1;
            // Get right children id from line
            next = line.find(delimiter, last);
            int right_children_id = std::stoi(line.substr(last, next - last));
            last = next + 1;
            // Get class name from line
            std::string class_name = line.substr(last);
            // Remove double quote characters around the class name
            class_name.erase(remove(class_name.begin(), class_name.end(), '"'), class_name.end());
            TreeNode new_tree = {class_name, threshold, feature_id, left_children_id, right_children_id};
            this->insert_node(node_id, new_tree);
        }
    }

    this->depth_up_to_date = false;
}

std::string DecisionTree::predict_class(const real_vector_t &features_vector) {
    // Verify that the tree does not contain a feature id bigger than the feature vector size
    if ((int) features_vector.size() < this->max_feature_id) {
        LOG(LOG_ERROR) << "Error : trying to make prediction but the tree contain a feature id bigger than the feature vector size (" << features_vector.size() << " < " << this->max_feature_id << ")";
        throw std::invalid_argument("Feature vector too small!");
    }

    // Current node point to root node
    TreeNode *current_node = &this->tree.at(0);

    while (current_node->as_children() && current_node->get_feature_id() >= 0) {
        // Set current node to the left or right node if the features vector feature at the given id is bellow the current node threshold
        if (features_vector.at(current_node->get_feature_id()) <= current_node->get_threshold()) {
            current_node = &this->tree.at(current_node->get_left_children_id());
        } else {
            current_node = &this->tree.at(current_node->get_right_children_id());
        }
    }

    return current_node->get_class_name();
}

/* PRIVATE DEFINITION */
std::size_t DecisionTree::compute_depth() const {
    // function <return_type(parameter_types)> function_name
    std::function<int(TreeNode)> compute_tree_depth = [this, &compute_tree_depth](const TreeNode &node) {
        int lh = (node.get_left_children_id() == -1) ? 0 : compute_tree_depth(this->tree.at(node.get_left_children_id()));
        int rh = (node.get_right_children_id() == -1) ? 0 : compute_tree_depth(this->tree.at(node.get_right_children_id()));
        return (std::size_t) std::max(lh, rh) + 1;
    };
    //TODO: parallelize the depth computation
    if (this->tree.empty()) {
        return 0;
    } else {
        return compute_tree_depth(this->tree.at(0));
    }
}









