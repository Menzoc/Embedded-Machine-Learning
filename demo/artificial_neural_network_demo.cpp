#include "../helpers/file_helpers.h"
#include "../helpers/print_helpers.h"
#include "../helpers/classification_helpers.h"
#include "../helpers/log.h"
#include "../ml_algorithms/artificial_neural_network.h"
log_struct LOGGING_CONFIG = {};

int main() {
    // Log config
    LOGGING_CONFIG.level = LOG_INFO;

    LOG(LOG_INFO) << "------------ Testing Artificial Neural Network using STFT algorithm ------------";
    // Get features from csv file
    LOG(LOG_INFO) << "Getting features vectors from the csv file " << absolute(MUSIC_FEATURES_STFT_CSV_TEST_PATH) << "...";
    auto fvs_stft = get_features_vectors_from_csv(MUSIC_FEATURES_STFT_CSV_TEST_PATH, AuFileProcessingAlgorithm::STFT);
    ArtificialNeuralNetwork artificial_neural_network_stft = {};
    // Create an artificial neural network from csv files
    LOG(LOG_INFO) << "Creating an Artificial Neural Network from all csv files in the following dir " << ARTIFICIAL_NEURAL_NETWORK_LAYERS_FOLDER_PATH_STFT << " ...";
    artificial_neural_network_stft.fill_from_csv(ARTIFICIAL_NEURAL_NETWORK_LAYERS_FOLDER_PATH_STFT);
    LOG(LOG_DEBUG) << "artificial neural network (" << artificial_neural_network_stft.get_layers().size() << " layers):\n" << artificial_neural_network_stft;

    // Predict and test prediction results
    auto predictions_stft = make_predictions(artificial_neural_network_stft, fvs_stft);
    auto prediction_accuracy_stft = predictions_report(predictions_stft);
    LOG(LOG_INFO) << "Model accuracy: "<< prediction_accuracy_stft;
    show_confusion_matrix(predictions_stft);

    LOG(LOG_INFO) << "------------ Testing Artificial Neural Network using MFCC algorithm ------------";
    // Get features from csv file
    LOG(LOG_INFO) << "Getting features vectors from the csv file " << absolute(MUSIC_FEATURES_MFCC_CSV_TEST_PATH) << " ...";
    auto fvs_mfcc = get_features_vectors_from_csv(MUSIC_FEATURES_MFCC_CSV_TEST_PATH, AuFileProcessingAlgorithm::MFCC);
    ArtificialNeuralNetwork artificial_neural_network_mfcc = {};
    // Create a tree from csv file
    LOG(LOG_INFO) << "Creating an Artificial Neural Network from all csv files in the following dir " << ARTIFICIAL_NEURAL_NETWORK_LAYERS_FOLDER_PATH_MFCC << " ...";
    artificial_neural_network_mfcc.fill_from_csv(ARTIFICIAL_NEURAL_NETWORK_LAYERS_FOLDER_PATH_MFCC);
    LOG(LOG_DEBUG) << "artificial neural network (" << artificial_neural_network_mfcc.get_layers().size() << " layers):\n" << artificial_neural_network_mfcc;

    // Predict and test prediction results
    auto predictions_mfcc = make_predictions(artificial_neural_network_mfcc, fvs_mfcc);
    auto prediction_accuracy_mfcc = predictions_report(predictions_mfcc);
    LOG(LOG_INFO) << "Model accuracy: "<< prediction_accuracy_mfcc;
    show_confusion_matrix(predictions_mfcc);

    return 0;
}
