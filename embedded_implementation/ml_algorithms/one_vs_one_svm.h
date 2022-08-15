
#ifndef ONE_VS_ONE_SVM_H
#define ONE_VS_ONE_SVM_H

#include "machine_learning_model.h"
#include "globals.h"
#include "../helpers/log.h"

/*
 * LinearClassifier class definition
 */

class LinearClassifier {
public:
    LinearClassifier(std::string lower_class, std::string upper_class, real_t intercept, real_vector_t coef_matrix);

    const std::string &get_lower_class() const;

    const std::string &get_upper_class() const;

    real_t get_intercept() const;

    const real_vector_t &get_coef_matrix() const;

    friend std::ostream &operator<<(std::ostream &os, const LinearClassifier &linear_classifier);

    std::string predict(const real_vector_t &features_vector);

private:
    std::string lower_class;
    std::string upper_class;
    real_t intercept;
    real_vector_t coeff_matrix;
};

/*
 * OneVsOneSVM class definition
 */
class OneVsOneSVM : public MachineLearningModel {

public:
    OneVsOneSVM();

    const std::vector<LinearClassifier> &get_classifiers() const;

    size_t get_number_of_class() const;

    friend std::ostream &operator<<(std::ostream &os, const OneVsOneSVM &one_vs_one_svm);

    void push_classifier(const LinearClassifier& linear_classifier);

    void pop_classifier();

    void clear();

    void fill_from_csv(const std::filesystem::path &csv_file_path) override;

    std::string predict_class(const real_vector_t &features_vector) override;


private:
    std::vector<LinearClassifier> classifiers;
    std::size_t number_of_class;

};

#endif //ONE_VS_ONE_SVM_H
