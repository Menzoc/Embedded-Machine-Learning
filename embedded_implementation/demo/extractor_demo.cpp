#include <chrono>
#include <tuple>
#include <execution>
#include "../helpers/file_helpers.h"
#include "../helpers/print_helpers.h"
#include "../helpers/log.h"
#include "../extraction/au_file_processor.h"


/** @brief extractor types of dataset */
enum class DatasetType : std::size_t {
    TRAIN = 0, /** The train dataset */
    TEST = 1 /** The test dataset */
};

/**
 * @brief           Extract all files from the given file list using a specified processing algorithm and dataset
 *
 * @param[in]       file_list a std::vector<std::filesystem::path> vector of all files to process
 * @param[in]       processing_algorithm a AuFileProcessingAlgorithm enum object designating the processing algorithm to use
 * @param[in]       dataset_type a DatasetType enum object designating the dataset to use
 * @returns         void
 */
void extract_files(const std::vector<std::filesystem::path> &file_list, AuFileProcessingAlgorithm processing_algorithm, DatasetType dataset_type) {
    std::filesystem::path csv_file_path;
    std::string processing_algorithm_string, dataset_type_string;

    // Parse parameters
    switch (dataset_type) {
        case DatasetType::TRAIN: {
            dataset_type_string = "TRAIN";
            switch (processing_algorithm) {
                case AuFileProcessingAlgorithm::STFT: {
                    csv_file_path = MUSIC_FEATURES_STFT_CSV_TRAIN_PATH;
                    processing_algorithm_string = "STFT";
                    break;
                }
                case AuFileProcessingAlgorithm::MFCC: {
                    csv_file_path = MUSIC_FEATURES_MFCC_CSV_TRAIN_PATH;
                    processing_algorithm_string = "MFCC";
                    break;
                }
                default: {
                    LOG(LOG_ERROR) << "Error : the processing algorithm value usage is not defined in the project";
                    throw std::domain_error("Unsupported processing algorithm!");
                    break;
                }
            }
            break;
        }
        case DatasetType::TEST: {
            dataset_type_string = "TEST";
            switch (processing_algorithm) {
                case AuFileProcessingAlgorithm::STFT: {
                    csv_file_path = MUSIC_FEATURES_STFT_CSV_TEST_PATH;
                    processing_algorithm_string = "STFT";
                    break;
                }
                case AuFileProcessingAlgorithm::MFCC: {
                    csv_file_path = MUSIC_FEATURES_MFCC_CSV_TEST_PATH;
                    processing_algorithm_string = "MFCC";
                    break;
                }
                default: {
                    LOG(LOG_ERROR) << "Error : the processing algorithm value usage is not defined in the project";
                    throw std::domain_error("Unsupported processing algorithm!");
                    break;
                }
            }
            break;
        }
        default: {
            LOG(LOG_ERROR) << "Error : the dataset type value usage is not defined in the project";
            throw std::domain_error("Unsupported dataset type!");
            break;
        }
    }

    // Open the csv file and write the header for the dataset
    std::ofstream csv_file(csv_file_path);
    csv_file << AuFileProcessor::get_csv_line_header(processing_algorithm);

    // Processing all files and writing processed features into the csv file
    LOG(LOG_INFO) << "------------ Processing files from the " << dataset_type_string << " dataset using the " << processing_algorithm_string << " algorithm ------------";
    auto start_time = std::chrono::high_resolution_clock::now();

    std::for_each(std::execution::seq, file_list.cbegin(), file_list.cend(), [&processing_algorithm, &csv_file](const std::filesystem::path& file) {
        try {
            LOG(LOG_DEBUG) << "Processing file " << file.filename().string() << "...";
            auto started_chrono = std::chrono::high_resolution_clock::now();
            AuFileProcessor au_file(file, processing_algorithm);
            au_file.read_file();
            LOG(LOG_DEBUG) << au_file.get_file_path().filename().string() << " details: " << au_file;
            auto stopped_chrono = std::chrono::high_resolution_clock::now();
            auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(stopped_chrono - started_chrono).count();
            LOG(LOG_DEBUG) << "File read in " << elapsed_time / 1000 << "s and " << elapsed_time % 1000 << "ms";
            started_chrono = std::chrono::high_resolution_clock::now();
            LOG(LOG_DEBUG) << "Applying processing algorithm on file " << file.filename().string() << " data...";
            au_file.apply_processing_algorithm();
            LOG(LOG_DEBUG) << au_file.get_file_path().filename().string() << " avg[" << au_file.get_features_average().size() << "]: " << au_file.get_features_average();
            LOG(LOG_DEBUG) << au_file.get_file_path().filename().string() << " std[" << au_file.get_features_standard_deviation().size() << "]: " << au_file.get_features_standard_deviation();
            stopped_chrono = std::chrono::high_resolution_clock::now();
            elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(stopped_chrono - started_chrono).count();
            LOG(LOG_DEBUG) << "File processed in " << elapsed_time / 1000 << "s and " << elapsed_time % 1000 << "ms";
            LOG(LOG_DEBUG) << "Adding file " << file.filename().string() << " features to csv file...";
            csv_file << au_file.get_csv_line() + "\n";
        } catch (const std::exception &e) {
            LOG(LOG_ERROR) << "File " << file.filename().string() << " not processed due to the following error : " << e.what();
        }
    });
    csv_file.close();
    LOG(LOG_INFO) << "Files features vector, music style and path writen in the CSV file " << absolute(csv_file_path);
    auto stop_time = std::chrono::high_resolution_clock::now();
    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(stop_time - start_time).count();
    LOG(LOG_INFO) << "------------ " << file_list.size() << " music files read in " << elapsed_time / 1000 << "s and " << elapsed_time % 1000 << "ms ------------";
}

log_struct LOGGING_CONFIG = {};

int main() {
    // log config
    LOGGING_CONFIG.level = LOG_INFO;

    auto dirs = alpha_dir_listing("../../../datasets/music");
    std::vector<std::filesystem::path> training_files;
    std::vector<std::filesystem::path> testing_files;

    // Select random files of each music style
    for (const auto &dir_path: dirs) {
        auto files = alpha_files_listing(dir_path);
        std::vector<std::filesystem::path> training;
        std::vector<std::filesystem::path> testing;
        std::tie(training, testing) = select_train_test_files(files, 0.2);
        training_files.insert(training_files.end(), training.begin(), training.end());
        testing_files.insert(testing_files.end(), testing.begin(), testing.end());
    }

    LOG(LOG_INFO) << "------------ Found " << training_files.size() << " training files ------------";
    LOG(LOG_INFO) << "------------ Found " << testing_files.size() << " testing files ------------";

    // Extracting training files using STFT
    extract_files(training_files, AuFileProcessingAlgorithm::STFT, DatasetType::TRAIN);
    // Extracting testing files using STFT
    extract_files(testing_files, AuFileProcessingAlgorithm::STFT, DatasetType::TEST);
    // Extracting training files using MFCC
    extract_files(training_files, AuFileProcessingAlgorithm::MFCC, DatasetType::TRAIN);
    // Extracting testing files using MFCC
    extract_files(testing_files, AuFileProcessingAlgorithm::MFCC, DatasetType::TEST);

    return 0;
}
