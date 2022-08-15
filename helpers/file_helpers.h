#ifndef FILE_HELPERS_H
#define FILE_HELPERS_H

#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <tuple>
#include <set>
#include <filesystem>
#include <map>
#include <random>
#include "globals.h"
#include "log.h"
#include "../extraction/au_file_processor.h"

static inline std::vector<std::filesystem::path> alpha_dir_listing(const std::string &dir_path) {
    std::set<std::filesystem::path> sorted_by_name_dirs;
    std::set<std::filesystem::path> sorted_by_name_files;
    std::vector<std::filesystem::path> dirs_listing;
    for (auto &entry: std::filesystem::directory_iterator(dir_path)) {
        if (std::filesystem::is_directory(entry.path())) {
            sorted_by_name_dirs.insert(entry.path());
        }
    }
    std::copy(sorted_by_name_dirs.cbegin(), sorted_by_name_dirs.cend(), std::back_inserter(dirs_listing));
    return dirs_listing;
}

static inline std::vector<std::filesystem::path> alpha_files_listing(const std::string &dir_path, const std::string &file_extension = "none") {
    std::set<std::filesystem::path> sorted_by_name_files;
    std::vector<std::filesystem::path> files_listing;
    for (const std::filesystem::path &file: std::filesystem::directory_iterator(dir_path)) {
        if (file_extension == "none") {
            sorted_by_name_files.insert(file);
        } else {
            if (file.extension() == file_extension) {
                sorted_by_name_files.insert(file);
            }
        }
    }
    std::copy(sorted_by_name_files.cbegin(), sorted_by_name_files.cend(), std::back_inserter(files_listing));
    return files_listing;
}

static inline std::pair<std::vector<std::filesystem::path>, std::vector<std::filesystem::path>>
select_train_test_files(std::vector<std::filesystem::path> files, double ratio) {
    std::size_t training_size = std::floor(files.size() * (1.0 - ratio));
    //std::size_t testing_size = files.size() - training_size;
    //std::cout << training_size << " " << testing_size << std::endl;
    std::random_device random_device;
    //std::mt19937 engine{66};
    std::mt19937 engine{random_device()};
    std::uniform_int_distribution<int> dist(0, files.size() - 1);
    std::set<std::filesystem::path> training_files_set;
    std::set<int> indexes;
    for (std::size_t k = 0; k < training_size; k++) {
        int random_index;
        do {
            random_index = dist(engine);
        } while (indexes.contains(random_index));
        indexes.insert(random_index);
        training_files_set.insert(files[random_index]);
    }
    std::vector<std::filesystem::path> testing_files;
    for (std::size_t k = 0; k < files.size(); k++) {
        if (!indexes.contains(k))
            testing_files.push_back(files[k]);
    }
    std::vector<std::filesystem::path> training_files;
    std::copy(training_files_set.cbegin(), training_files_set.cend(), std::back_inserter(training_files));
    return std::make_pair(training_files, testing_files);
}

static inline std::vector<std::pair<std::string, real_vector_t>> get_features_vectors_from_csv(const std::filesystem::path &csv_file_path, const AuFileProcessingAlgorithm &processing_algorithm) {
    std::vector<std::pair<std::string, real_vector_t>> features_vector_list = {};

    // Get data length depending on processing algorithm
    std::size_t data_length;
    switch (processing_algorithm) {
        case AuFileProcessingAlgorithm::STFT: {
            // The data length for the STFT algorithm is FFT_SIZE*(2->avg+stdev)
            data_length = FFT_SIZE * 2;
            break;
        }
        case AuFileProcessingAlgorithm::MFCC: {
            // The data length for the MFCC algorithm is [MEL_APPLIED_N+ (1->energy)] * (2->avg+stdev)
            data_length = (MEL_APPLIED_N + 1) * 2;
            break;
        }
        default: {
            LOG(LOG_ERROR) << "Error : the processing algorithm value usage is not defined in the project";
            throw std::domain_error("Unsupported processing algorithm!");
            break;
        }
    }

    const char delimiter = ',';
    std::string line = {};
    std::string data = {};

    std::ifstream input_file(csv_file_path);
    if (!input_file.is_open()) {
        LOG(LOG_ERROR) << "Error : file with path " + csv_file_path.string() + "  not found.";
        throw std::filesystem::filesystem_error("Can't open file!", std::make_error_code(std::errc::no_such_file_or_directory));
    }

    bool header_skipped = false;
    while (std::getline(input_file, line)) {
        if (!header_skipped) {
            header_skipped = true;
            continue;
        } else {
            std::stringstream ss(line);
            size_t last = 0;
            size_t next = 0;

            // Get all values (data_length first data of csv line)
            std::vector<real_t> new_fv = {};
            for (std::size_t i = 0; i < data_length; i++) {
                next = line.find(delimiter, last);
                new_fv.push_back((real_t) std::stod(line.substr(last, next - last)));
                last = next + 1;
            }
            // Get the music style (first data after the N values of csv line)
            next = line.find(delimiter, last);
            std::string new_fv_style = line.substr(last, next - last);
            // Remove double quote characters around the music style
            new_fv_style.erase(remove(new_fv_style.begin(), new_fv_style.end(), '"'), new_fv_style.end());

            features_vector_list.emplace_back(new_fv_style, new_fv);
        }
    }

    return features_vector_list;
}


#endif //FILE_HELPERS_H
