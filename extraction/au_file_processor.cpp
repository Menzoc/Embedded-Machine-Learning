#include "au_file_processor.h"
#include "../helpers/log.h"
#include "../helpers/signal.h"
#include "../helpers/print_helpers.h"

#include <fstream>

/* PUBLIC DEFINITION */
AuFileProcessor::AuFileProcessor(const std::filesystem::path &file_path) : file_path(file_path) {
    // Testing file extension
    if (file_path.filename().extension() != ".au") {
        LOG(LOG_ERROR) << "Error : file " << file_path.filename() << " does not have the .au extension";
        throw std::domain_error("Not a .au file!");
    }

    // Set the processing algorithm to the default value
    this->processing_algorithm = AuFileProcessor::DEFAULT_PROCESSING_ALGORITHM;

    // Extracting music style from file name
    std::string string_file_path = file_path.filename().string();
    this->music_style = string_file_path.substr(0, string_file_path.find('.'));

    this->magic_number = 0;
    this->data_offset = 0;
    this->data_size = 0;
    this->encoding = 0;
    this->sample_rate = 0;
    this->channels = 0;
    this->features_average = {};
    this->features_standard_deviation = {};

}

AuFileProcessor::AuFileProcessor(const std::filesystem::path &file_path, AuFileProcessingAlgorithm processing_algorithm) : file_path(file_path), processing_algorithm(processing_algorithm) {
    // Testing file extension
    if (file_path.filename().extension() != ".au") {
        LOG(LOG_ERROR) << "Error : file " << file_path.filename() << " does not have the .au extension";
        throw std::domain_error("Not a .au file!");
    }

    // Extracting music style from file name
    std::string string_file_path = file_path.filename().string();
    this->music_style = string_file_path.substr(0, string_file_path.find('.'));

    this->magic_number = 0;
    this->data_offset = 0;
    this->data_size = 0;
    this->encoding = 0;
    this->sample_rate = 0;
    this->channels = 0;
    this->features_average = {};
    this->features_standard_deviation = {};
}

const std::filesystem::path &AuFileProcessor::get_file_path() const {
    return file_path;
}

AuFileProcessingAlgorithm AuFileProcessor::get_processing_algorithm() const {
    return processing_algorithm;
}

void AuFileProcessor::set_processing_algorithm(AuFileProcessingAlgorithm algorithm) {
    this->processing_algorithm = algorithm;
}

const std::string &AuFileProcessor::get_music_style() const {
    return music_style;
}

word_t AuFileProcessor::get_magic_number() const {
    return magic_number;
}

word_t AuFileProcessor::get_data_offset() const {
    return data_offset;
}

word_t AuFileProcessor::get_data_size() const {
    return data_size;
}

word_t AuFileProcessor::get_encoding() const {
    return encoding;
}

word_t AuFileProcessor::get_sample_rate() const {
    return sample_rate;
}

word_t AuFileProcessor::get_channels() const {
    return channels;
}

const real_vector_t &AuFileProcessor::get_raw_data() const {
    return raw_data;
}

const real_vector_t &AuFileProcessor::get_features_average() const {
    return features_average;
}

const real_vector_t &AuFileProcessor::get_features_standard_deviation() const {
    return features_standard_deviation;
}

std::ostream &operator<<(std::ostream &os, const AuFileProcessor &au_file_processor) {
    char magic_str[5] = {
            (char) ((au_file_processor.magic_number & 0xFF000000) >> 24u),
            (char) ((au_file_processor.magic_number & 0x00FF0000) >> 16u),
            (char) ((au_file_processor.magic_number & 0x0000FF00) >> 8u),
            (char) ((au_file_processor.magic_number & 0x000000FF) >> 0u),
            '\0'
    };
    std::string data_size_human_readable = {};
    std::string size_unit;
    if (au_file_processor.data_size < KiB) {
        data_size_human_readable = std::to_string(au_file_processor.data_size) + "B";
    } else if (au_file_processor.data_size < MiB) {
        data_size_human_readable = std::to_string(au_file_processor.data_size / KiB) + "." + std::to_string(au_file_processor.data_size % KiB) + "KiB";
    } else if (au_file_processor.data_size < GiB) {
        data_size_human_readable = std::to_string(au_file_processor.data_size / MiB) + "." + std::to_string(au_file_processor.data_size % MiB) + "MiB";
    } else {
        data_size_human_readable = std::to_string(au_file_processor.data_size / GiB) + "." + std::to_string(au_file_processor.data_size % GiB) + "GiB";
    }

    return os << std::hex << "{magic number: 0x" << au_file_processor.magic_number << " (" << magic_str << ")"
              << std::dec << ", data offset: " << au_file_processor.data_offset
              << std::dec << ", data size: " << data_size_human_readable
              << std::dec << ", encoding: " << au_file_processor.encoding
              << std::dec << ", sample rate (sample/sec): " << au_file_processor.sample_rate
              << std::dec << ", channels: " << au_file_processor.channels
              << "}";
}

void AuFileProcessor::read_file() {
    // Start read timer
    auto start_time = std::chrono::high_resolution_clock::now();

    // Opening audio file
    std::ifstream audio_file(this->file_path);
    if (!audio_file.is_open()) {
        LOG(LOG_ERROR) << "Error : file with path " + this->file_path.string() + "  not found.";
        throw std::filesystem::filesystem_error("Can't open file!", std::make_error_code(std::errc::no_such_file_or_directory));
    }
    // Processing file header
    this->magic_number = this->get_next_word(audio_file);
    if (this->magic_number != 0x2e736e64) {
        LOG(LOG_ERROR) << "Error : magic number does not match the required, the file encoding is not following the au file format";
        throw std::domain_error("Bad file encoding!");
    }
    this->data_offset = this->get_next_word(audio_file);
    this->data_size = this->get_next_word(audio_file);
    this->encoding = this->get_next_word(audio_file);
    this->sample_rate = this->get_next_word(audio_file);
    this->channels = this->get_next_word(audio_file);
    this->read_raw_data(audio_file);

    // Stop timer
    auto stop_time = std::chrono::high_resolution_clock::now();

    if (audio_file.is_open()) {
        audio_file.close();
    }
    LOG(LOG_DEBUG) << "time spent to read file " << this->file_path.filename() << ": " << (stop_time - start_time) / std::chrono::milliseconds(1) << "ms";
}

std::string AuFileProcessor::get_csv_line() {
    std::string csv_line;
    std::string features_average_csv_string = {};
    for (const real_t &r: this->features_average) {
        features_average_csv_string += (std::to_string(r) + ",");
    }
    std::string features_standard_deviation_csv_string = {};
    for (const real_t &r: this->features_standard_deviation) {
        features_standard_deviation_csv_string += (std::to_string(r) + ",");
    }
    csv_line = std::string(features_average_csv_string + features_standard_deviation_csv_string + "\"" + this->music_style + "\",\"" + this->file_path.string() + "\"");

    return csv_line;
}

std::string AuFileProcessor::get_csv_line_header(AuFileProcessingAlgorithm processing_algorithm) {
    std::string header = {};
    switch (processing_algorithm) {
        case AuFileProcessingAlgorithm::STFT: {
            for (std::size_t i = 0; i < FFT_SIZE; i++) {
                header += ("BIN_AVG" + std::to_string(i) + ",");
            }
            for (std::size_t i = 0; i < FFT_SIZE; i++) {
                header += ("BIN_STDEV" + std::to_string(i) + ",");
            }
            header += ("style,");
            header += ("file_name\n");
            break;
        }
        case AuFileProcessingAlgorithm::MFCC: {
            header += ("SIGNALENERGY_AVG,");
            for (std::size_t i = 0; i < MEL_APPLIED_N; i++) {
                header += ("BIN_AVG" + std::to_string(i) + ",");
            }
            header += ("SIGNALENERGY_STDEV,");
            for (std::size_t i = 0; i < MEL_APPLIED_N; i++) {
                header += ("BIN_STDEV" + std::to_string(i) + ",");
            }
            header += ("style,");
            header += ("file_name\n");
            break;
        }
        default: {
            LOG(LOG_ERROR) << "Error : the processing algorithm value usage is not defined in the project";
            throw std::domain_error("Unsupported processing algorithm!");
            break;
        }
    }
    return header;
}

void AuFileProcessor::apply_processing_algorithm() {
    // Apply process algorithm
    switch (this->processing_algorithm) {
        case AuFileProcessingAlgorithm::STFT: {
            apply_stft();
            break;
        }
        case AuFileProcessingAlgorithm::MFCC: {
            apply_mfcc();
            break;
        }
        default: {
            LOG(LOG_ERROR) << "Error : the processing algorithm value usage is not defined in the project";
            throw std::domain_error("Unsupported processing algorithm!");
            break;
        }
    }

    // Normalize features
    this->normalize_features();
}

/* PRIVATE DEFINITION */
word_t AuFileProcessor::get_next_word(std::ifstream &file) {
    word_t word = 0;
    uint8_t byte;
    // Read a word (unsigned 32 bit) byte per byte and store it in big endian
    for (std::size_t i = 0; i < 4; i++) {
        file.read(reinterpret_cast<char *>(&byte), 1);
        word = word | (byte << (((4 - 1) - i) * 8));
    }
    return word;
}

int16_t AuFileProcessor::get_next_data(std::ifstream &file) {
    int16_t data = 0;
    uint8_t byte;
    // Read a data (signed 16 bit) byte per byte and store it in big endian
    for (std::size_t i = 0; i < 2; i++) {
        file.read(reinterpret_cast<char *>(&byte), 1);
        data = data | (byte << (((2 - 1) - i) * 8));
    }
    return data;
}

void AuFileProcessor::read_raw_data(std::ifstream &file) {
    file.seekg(this->data_offset, std::ios_base::beg);
    std::size_t bytes_number = 0;
    switch (this->encoding) {
        case static_cast<word_t>(AuFileEncodingFormat::PCM_16B) : {
            bytes_number = 2;
            for (size_t i = 0; i < this->data_size / bytes_number; i++) {
                this->raw_data.push_back((real_t) this->get_next_data(file));
            }
            break;
        }
        default: {
            LOG(LOG_ERROR) << "Error : the encoding mode value (" << this->encoding << ") usage is not defined in the project";
            throw std::domain_error("Unsupported data encoding!");
            break;
        }
    }

    raw_data.shrink_to_fit();
}

void AuFileProcessor::apply_stft() {
    // Create hamming window
    real_n_array_t h_window = hamming_window();

    real_t size = 0;
    real_t x = 0;
    real_fft_array_t mean_prev = {};
    real_fft_array_t mean = {};
    real_fft_array_t stdv = {};
    this->features_average = {};
    this->features_standard_deviation = {};

    for (std::size_t k = 0; k < raw_data.size() / N; k++) {
        complex_n_array_t v1;
        complex_n_array_t v2;

        // insert raw data into v1 using a step of size N
        std::vector<complex_t> V1;
        std::copy(raw_data.cbegin() + k * N, raw_data.cbegin() + k * N + N, v1.begin());
        // insert raw data into v2 using a step of size N/2
        std::copy(raw_data.cbegin() + k * N + N / 2, raw_data.cbegin() + k * N + N + N / 2, v2.begin());

        // apply windowing and fft to v1 and v2
        windowing(h_window, v1);
        windowing(h_window, v2);
        // compute the fft of v1 and v2
        ite_dit_fft(v1);
        ite_dit_fft(v2);

        // We need only half of v1 and v2 because the fft result is symmetrical regarding the origin
        size++;
        for (std::size_t iter = 0; iter < FFT_SIZE; iter++) {
            x = std::abs(v1.at(iter));
            mean_prev.at(iter) = mean.at(iter);
            mean.at(iter) += (x - mean.at(iter)) / size;
            stdv.at(iter) += (x - mean.at(iter)) * (x - mean_prev.at(iter));
            mean_prev.at(iter) = mean.at(iter);
            x = std::abs(v2.at(iter));
            mean.at(iter) += (x - mean.at(iter)) / size;
            stdv.at(iter) += (x - mean.at(iter)) * (x - mean_prev.at(iter));
        }

    }
    this->features_average.clear();
    std::move(mean.cbegin(), mean.cend(), std::back_inserter(this->features_average));
    this->features_standard_deviation.clear();
    std::transform(stdv.begin(), stdv.end(), std::back_inserter(this->features_standard_deviation), [size](real_t c) { return std::sqrt(c / (size - 1)); });
}

void AuFileProcessor::apply_mfcc() {
    // Create bank of filter
    std::array<real_fft_array_t, MEL_N> filter_bank_t = mfcc_filters();
    std::array<real_fft_array_t, MEL_APPLIED_N> filter_bank = {};
    std::move(filter_bank_t.cbegin(), filter_bank_t.cbegin() + filter_bank.size(), filter_bank.begin());

    // Apply hamming window
    real_n_array_t h_window = hamming_window();
    real_vector_t energy;

    real_t size = 0;
    // Use an array of size MEL_APPLIED_N+1 because we have the filters and the energy
    std::array<real_t, MEL_APPLIED_N + 1> mean_prev = {};
    std::array<real_t, MEL_APPLIED_N + 1> mean = {};
    std::array<real_t, MEL_APPLIED_N + 1> stdv = {};
    this->features_average = {};
    this->features_standard_deviation = {};

    for (std::size_t k = 0; k < raw_data.size() / N; k++) {
        complex_n_array_t v1;
        complex_n_array_t v2;
        real_t energy_v1;
        real_t energy_v2;
        real_fft_array_t e1;
        real_fft_array_t e2;

        std::array<real_t, MEL_APPLIED_N + 1> e1_result{};
        std::array<real_t, MEL_APPLIED_N + 1> e2_result{};

        // insert raw data into v1 using a step of size N
        std::copy(raw_data.cbegin() + k * N, raw_data.cbegin() + k * N + N, v1.begin());
        // insert raw data into v2 using a step of size N/2
        std::copy(raw_data.cbegin() + k * N + N / 2, raw_data.cbegin() + k * N + N + N / 2, v2.begin());

        // apply windowing on v1 and v2
        windowing(h_window, v1);
        windowing(h_window, v2);

        // Compute frame energy (divide by 1e3 to be in the same order of magnitude as the means and standard deviation)
        energy_v1 = std::transform_reduce(v1.cbegin(), v1.cend(), 0.0, std::plus<>(), [](complex_t c)mutable {
            return std::log(std::max(std::abs(c * c), 2e-22)) / 1e3;
        });
        energy_v2 = std::transform_reduce(v2.cbegin(), v2.cend(), 0.0, std::plus<>(), [](complex_t c)mutable {
            return std::log(std::max(std::abs(c * c), 2e-22)) / 1e3;

        });

        // compute the fft of v1 and v2
        ite_dit_fft(v1);
        ite_dit_fft(v2);

        // We need only half of v1 and v2 because the fft result is symmetrical regarding the origin
        // note that function send directly the magnitude of each frequency
        // convert complex data to simple
        std::transform(std::execution::seq, v1.cbegin(), v1.cbegin() + (FFT_SIZE - 1), e1.begin(), [](complex_t c) { return std::abs(sqrt(c.real() * c.real() + c.imag() * c.imag())); });
        std::transform(std::execution::seq, v2.cbegin(), v2.cbegin() + (FFT_SIZE - 1), e2.begin(), [](complex_t c) { return std::abs(sqrt(c.real() * c.real() + c.imag() * c.imag())); });

        e1_result.at(0) = energy_v1;
        e2_result.at(0) = energy_v2;

        // Apply the dtc and filterbank on e1 and e2 (output vector size is MEL_APPLIED_N)
        // The function apply_filterbank take the log of each value -> non linear rectification
        real_vector_t e1_filt = dct2(apply_filterbank(filter_bank, e1));
        real_vector_t e2_filt = dct2(apply_filterbank(filter_bank, e2));

        std::move(e1_filt.begin(), e1_filt.end(), e1_result.begin() + 1);
        std::move(e2_filt.begin(), e2_filt.end(), e2_result.begin() + 1);

        size++;
        for (std::size_t iter = 0; iter < e1_result.size(); iter++) {
            mean_prev.at(iter) = mean.at(iter);
            mean.at(iter) += (e1_result.at(iter) - mean.at(iter)) / size;
            stdv.at(iter) += (e1_result.at(iter) - mean.at(iter)) * (e1_result.at(iter) - mean_prev.at(iter));
            mean_prev.at(iter) = mean.at(iter);
            mean.at(iter) += (e2_result.at(iter) - mean.at(iter)) / size;
            stdv.at(iter) += (e2_result.at(iter) - mean.at(iter)) * (e2_result.at(iter) - mean_prev.at(iter));
        }
    }

    this->features_average.clear();
    std::move(mean.cbegin(), mean.cend(), std::back_inserter(this->features_average));
    this->features_standard_deviation.clear();
    std::transform(stdv.begin(), stdv.end(), std::back_inserter(this->features_standard_deviation), [size](real_t c) { return std::sqrt(c / (size - 1)); });
}

void AuFileProcessor::normalize_features() {
    real_t features_size = (real_t) (this->features_average.size() + this->features_standard_deviation.size());

    // Compute features vector mean
    real_t sum = 0;
    std::for_each(std::execution::seq, this->features_average.cbegin(), this->features_average.cend(), [&sum](real_t x) mutable {
        sum += x;
    });
    std::for_each(std::execution::seq, this->features_standard_deviation.cbegin(), this->features_standard_deviation.cend(), [&sum](real_t x) mutable {
        sum += x;
    });
    real_t mean = sum / features_size;

    // Compute features vector standard deviation
    sum = 0;
    std::for_each(std::execution::seq, this->features_average.cbegin(), this->features_average.cend(), [&mean, &sum](real_t x) mutable {
        sum += std::pow(x - mean, 2);
    });
    std::for_each(std::execution::seq, this->features_standard_deviation.cbegin(), this->features_standard_deviation.cend(), [&mean, &sum](real_t x) mutable {
        sum += std::pow(x - mean, 2);
    });
    real_t stdev = std::sqrt(sum / features_size);

    // Normalize features_average
    std::transform(std::execution::seq, features_average.cbegin(), features_average.cend(), features_average.begin(), [&mean, &stdev](real_t x) {
        return (x - mean) / stdev;
    });
    // Normalize features_standard_deviation
    std::transform(std::execution::seq, features_standard_deviation.cbegin(), features_standard_deviation.cend(), features_standard_deviation.begin(), [&mean, &stdev](real_t x) {
        return (x - mean) / stdev;
    });

}
