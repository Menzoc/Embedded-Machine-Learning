#include "one_vs_one_svm.h"
#include <fstream>
#include <utility>
#include <execution>
#include <map>
#include "../helpers/print_helpers.h"

/*
 * LinearClassifier class definition
 */

/* PUBLIC DEFINITION */
LinearClassifier::LinearClassifier(std::string lower_class, std::string upper_class, real_t intercept, real_vector_t coef_matrix) :
        lower_class(std::move(lower_class)), upper_class(std::move(upper_class)), intercept(intercept), coeff_matrix(std::move(coef_matrix)) {
}

const std::string &LinearClassifier::get_lower_class() const {
    return lower_class;
}

const std::string &LinearClassifier::get_upper_class() const {
    return upper_class;
}

real_t LinearClassifier::get_intercept() const {
    return intercept;
}

const real_vector_t &LinearClassifier::get_coef_matrix() const {
    return coeff_matrix;
}

std::ostream &operator<<(std::ostream &os, const LinearClassifier &linear_classifier) {
    // A for_each can be used here because the size of the matrix_a string will not be too long
    size_t coef_matrix_len = linear_classifier.get_coef_matrix().size();
    std::string matrix_a = "[";
    std::for_each(std::execution::seq, linear_classifier.get_coef_matrix().cbegin(), linear_classifier.get_coef_matrix().cend(), [&matrix_a, &coef_matrix_len, i = 0](real_t r)mutable {
        matrix_a += (i < int(coef_matrix_len)) ? std::to_string(r) + ", " : std::to_string(r);
    });
    matrix_a += "]";
    return os << "y >= Ax+b -> class id is " << linear_classifier.get_upper_class() << " else class id is" << linear_classifier.get_lower_class() << " with A=" << matrix_a << " and b=" << std::to_string(linear_classifier.get_intercept());
}

std::string LinearClassifier::predict(const real_vector_t &features_vector) {
    LOG(LOG_DEBUG) << "fv: " << features_vector;
    LOG(LOG_DEBUG) << "intercept: " << intercept << ", coeff_matrix: " << this->coeff_matrix;
    if (features_vector.size() != this->coeff_matrix.size()) {
        LOG(LOG_ERROR) << "Error : trying to make prediction but the feature vector size (" << features_vector.size() << ") is different from the coefficient matrix size (" << this->coeff_matrix.size() << ")";
        throw std::invalid_argument("Feature vector size differ from coefficient matrix size!");
    }

    real_t result = 0;
    std::for_each(std::execution::seq, this->coeff_matrix.cbegin(), this->coeff_matrix.cend(), [&features_vector, &result, i = 0](real_t x) mutable {
        result += features_vector.at(i) * x;
        i++;
    });
    result += this->intercept;

    LOG(LOG_DEBUG) << this->upper_class << " (" << result << ") " << this->lower_class;
    return (result > 0 ? this->lower_class : this->upper_class);
}

/* PRIVATE DEFINITION */


/*
 * OneVsOneSVM class definition
 */

/* PUBLIC DEFINITION */
OneVsOneSVM::OneVsOneSVM() {
    this->classifiers.clear();
    this->number_of_class = 0;
}

const std::vector<LinearClassifier> &OneVsOneSVM::get_classifiers() const {
    return classifiers;
}

size_t OneVsOneSVM::get_number_of_class() const {
    return number_of_class;
}

std::ostream &operator<<(std::ostream &os, const OneVsOneSVM &one_vs_one_svm) {
    // A for_each is not used here because the size of the final string can be really long
    os << "One Vs One Linear SVM classifiers:" << std::endl;
    std::size_t i = 0;
    for (const LinearClassifier &c: one_vs_one_svm.get_classifiers()) {
        os << "\t[" << i << "] " << c << std::endl;
        i++;
    }
    return os;
}

void OneVsOneSVM::push_classifier(const LinearClassifier &linear_classifier) {
    this->classifiers.push_back(linear_classifier);
}

void OneVsOneSVM::pop_classifier() {
    this->classifiers.pop_back();
}

void OneVsOneSVM::clear() {
    this->classifiers.clear();
}

void OneVsOneSVM::fill_from_csv(const std::filesystem::path &csv_file_path) {
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
            // Get class if prediction result is positive from line
            next = line.find(delimiter, last);
            std::string positive_class = line.substr(last, next - last);
            // Remove double quote characters around the class name
            positive_class.erase(remove(positive_class.begin(), positive_class.end(), '"'), positive_class.end());
            last = next + 1;
            // Get class if prediction result is negative from line
            next = line.find(delimiter, last);
            std::string negative_class = line.substr(last, next - last);
            // Remove double quote characters around the class name
            negative_class.erase(remove(negative_class.begin(), negative_class.end(), '"'), negative_class.end());
            last = next + 1;
            // Get the intercept from line
            next = line.find(delimiter, last);
            real_t intercept = (real_t) std::stod(line.substr(last, next - last));
            last = next + 1;
            // Get the coeff matrix from line
            real_vector_t coeff_matrix;
            while ((next = line.find(delimiter, last)) != std::string::npos) {
                coeff_matrix.push_back((real_t) std::stod(line.substr(last, next - last)));
                last = next + 1;
            }
            coeff_matrix.push_back((real_t) std::stod(line.substr(last)));

            LinearClassifier new_linear_classifier = {positive_class, negative_class, intercept, coeff_matrix};
            this->push_classifier(new_linear_classifier);
        }
    }
}

std::string OneVsOneSVM::predict_class(const real_vector_t &features_vector) {
    std::map<std::string, std::size_t> pred_results;
    using pred_results_pair_t = decltype(pred_results)::value_type;

    // Get results for each linear classifier
    std::for_each(std::execution::seq, this->classifiers.begin(), this->classifiers.end(), [&pred_results, &features_vector](LinearClassifier &linear_classifier) mutable {
        std::string pred_result = linear_classifier.predict(features_vector);
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
