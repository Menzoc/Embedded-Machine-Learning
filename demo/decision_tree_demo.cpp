#include <chrono>
#include "../helpers/file_helpers.h"
#include "../helpers/print_helpers.h"
#include "../helpers/log.h"
#include "../helpers/classification_helpers.h"
#include "../ml_algorithms/decision_tree.h"

log_struct LOGGING_CONFIG = {};

int main() {
    // Log config
    LOGGING_CONFIG.level = LOG_INFO;

    LOG(LOG_INFO) << "------------ Testing Decision Tree using STFT algorithm ------------";
    // Get features from csv file
    LOG(LOG_INFO) << "Getting features vectors from the csv file " << absolute(MUSIC_FEATURES_STFT_CSV_TEST_PATH) << "...";
    auto fvs_stft = get_features_vectors_from_csv(MUSIC_FEATURES_STFT_CSV_TEST_PATH, AuFileProcessingAlgorithm::STFT);
    DecisionTree decision_tree_model_stft = {};
    // Create a tree from csv file
    LOG(LOG_INFO) << "Creating a Decision Tree from the csv file " << DECISION_TREE_CSV_PATH_STFT << " ...";
    decision_tree_model_stft.fill_from_csv(DECISION_TREE_CSV_PATH_STFT);
    LOG(LOG_DEBUG) << "decision tree (depth: " << decision_tree_model_stft.get_depth() << "): " << decision_tree_model_stft;

    // Predict and test prediction results
    auto predictions_stft = make_predictions(decision_tree_model_stft, fvs_stft);
    auto prediction_accuracy_stft = predictions_report(predictions_stft);
    LOG(LOG_INFO) << "Model accuracy: "<< prediction_accuracy_stft;
    show_confusion_matrix(predictions_stft);


    LOG(LOG_INFO) << "------------ Testing Decision Tree using MFCC algorithm ------------";
    // Get features from csv file
    LOG(LOG_INFO) << "Getting features vectors from the csv file " << absolute(MUSIC_FEATURES_MFCC_CSV_TEST_PATH) << " ...";
    auto fvs_mfcc = get_features_vectors_from_csv(MUSIC_FEATURES_MFCC_CSV_TEST_PATH, AuFileProcessingAlgorithm::MFCC);
    DecisionTree decision_tree_model_mfcc = {};
    // Create a tree from csv file
    LOG(LOG_INFO) << "Creating a Decision Tree from the csv file " << DECISION_TREE_CSV_PATH_MFCC << " ...";
    decision_tree_model_mfcc.fill_from_csv(DECISION_TREE_CSV_PATH_MFCC);
    LOG(LOG_DEBUG) << "decision tree (depth: " << decision_tree_model_mfcc.get_depth() << "): " << decision_tree_model_mfcc;

    // Predict and test prediction results
    auto predictions_mfcc = make_predictions(decision_tree_model_mfcc, fvs_mfcc);
    auto prediction_accuracy_mfcc = predictions_report(predictions_mfcc);
    LOG(LOG_INFO) << "Model accuracy: "<< prediction_accuracy_mfcc;
    show_confusion_matrix(predictions_mfcc);

    return 0;
}
