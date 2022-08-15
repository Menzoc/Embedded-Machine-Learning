#ifndef AU_FILE_PROCESSOR
#define AU_FILE_PROCESSOR

#include "globals.h"
#include <filesystem>
#include <vector>
#include <complex>

/** @brief .au file encoding formats */
enum class AuFileEncodingFormat : word_t {
    MICRO_8B = 1, /** 8-bit G.711 μ-law */
    PCM_8B = 2, /** 8-bit linear PCM */
    PCM_16B = 3, /** 16-bit linear PCM */
    PCM_24B = 4, /** 24-bit linear PCM */
    PCM_32B = 5, /** 32-bit linear PCM */
    IEEE_32B = 6, /** 32-bit IEEE floating point */
    IEEE_64B = 7, /** 64-bit IEEE floating point */
    FRAGMENTED = 8, /** Fragmented sample data */
    DSP_PROGRAM = 9, /** DSP program */
    FIXED_8B = 10, /** 8-bit fixed point */
    FIXED_16B = 11, /** 16-bit fixed point */
    FIXED_24B = 12, /** 24-bit fixed point */
    FIXED_32B = 13, /** 32-bit fixed point */
    LINEAR_EMPHASIS_16B = 18, /** 16-bit linear with emphasis */
    LINEAR_COMPRESSES_16B = 19, /** 16-bit linear compressed */
    LINEAR_EMPHASIS_COMPRESSED_16B = 20, /** 16-bit linear with emphasis and compression */
    MUSIC_KIT_DSP_PROGRAM = 21, /** Music kit DSP commands */
    ITUT_G721_ADPCM_4B = 23, /** 4-bit compressed using the ITU-T G.721 ADPCM voice data encoding scheme */
    ITUT_G722_SB_ADPCM_4B = 24, /** ITU-T G.722 SB-ADPCM */
    ITUT_G723_ADPCM_3B = 25, /** ITU-T G.723 3-bit ADPCM */
    ITUT_G723_ADPCM_5B = 26, /** ITU-T G.723 5-bit ADPCM */
    ALAX_G711_8B = 27 /** 8-bit G.711 A-law */
};

/** @brief .au file process methods */
enum class AuFileProcessingAlgorithm : std::size_t {
    STFT = 0, /** Process the data using only the STFT algorithm */
    MFCC = 1 /** Process the data using only the MFCC algorithm */
};

/*
 * AuFileProcessor class definition
 */
class AuFileProcessor {
public:
    /**
     * @brief           AuFileProcessor constructor.
     *
     * @param[in]       file_path a std::filesystem::path object of the .au file to process
     * @trhow           <std::domain_error("Not a .au file!")> Throw an exception if the file extension is not .au
     * @returns         the constructed object
     */
    explicit AuFileProcessor(const std::filesystem::path &file_path);

    /**
    * @brief           AuFileProcessor constructor.
    *
    * @param[in]       file_path a std::filesystem::path object of the .au file to process
    * @param[in]       processing_algorithm a AuFileProcessingAlgorithm enum variable that represent the processing algorithm to use
    * @trhow           <std::domain_error("Not a .au file!")> Throw an exception if the file extension is not .au
    * @returns         the constructed object
    */
    explicit AuFileProcessor(const std::filesystem::path &file_path, AuFileProcessingAlgorithm processing_algorithm);

    /**
     * @brief           Get the .au file path.
     *
     * @returns         the file_path class variable value.
     */
    const std::filesystem::path &get_file_path() const;

    /**
     * @brief           Get the .au file current process algorithm.
     *
     * @returns         the processing_algorithm class variable value.
     */
    AuFileProcessingAlgorithm get_processing_algorithm() const;

    /**
     * @brief           Set the .au current process algorithm.
     * param[in]        processing_algorithm a AuFileProcessingAlgorithm enum variable that represent the process algorithm to use
     *
     * @returns         void
     */
    void set_processing_algorithm(AuFileProcessingAlgorithm algorithm);

    /**
     * @brief           Get the .au file music style.
     *
     * @returns         the music_style class variable value.
     */
    const std::string &get_music_style() const;

    /**
     * @brief           Get the .au file magic number.
     *
     * @returns         the magic_number class variable value.
     */
    word_t get_magic_number() const;

    /**
     * @brief           Get the .au file data offset.
     *
     * @returns         the data_offset class variable value.
     */
    word_t get_data_offset() const;

    /**
     * @brief           Get the .au file data size.
     *
     * @returns         the data_size class variable value.
     */
    word_t get_data_size() const;

    /**
     * @brief           Get the .au file encoding format.
     *
     * @returns         the encoding class variable value.
     */
    word_t get_encoding() const;

    /**
     * @brief           Get the .au file sample rate (in sample/sec).
     *
     * @returns         the sample_rate class variable value.
     */
    word_t get_sample_rate() const;

    /**
     * @brief           Get the .au file number of interleaved channels.
     *
     * @returns         the channels class variable value.
     */
    word_t get_channels() const;

    /**
     * @brief           Get the .au file raw data vector.
     *
     * @returns         the raw_data class variable value.
     */
    const real_vector_t &get_raw_data() const;

    /**
     * @brief           Get the .au file features average.
     *
     * @returns         the features_average class variable value.
     */
    const real_vector_t &get_features_average() const;

    /**
     * @brief           Get the .au file features standard deviation.
     *
     * @returns         the features_standard_deviation class variable value.
     */
    const real_vector_t &get_features_standard_deviation() const;

    /**
     * @brief           Overload the << operator for the AuFileProcessor object.
     *
     * @returns         the ostream of the human readable object
     */
    friend std::ostream &operator<<(std::ostream &os, const AuFileProcessor &au_file_processor);

    /**
     * @brief           Parse the .au file header and data and save them to their corresponding class variables.
     * @details         The time spent to parse all the file is written in LOG_DEBUG level.
     *
     * @trhow           <filesystem_error("Can't open file!", std::make_error_code(std::errc::no_such_file_or_directory))> Throw an exception if the file cannot be opened
     * @trhow           <std::domain_error("Bad file encoding!")> Throw an exception if the magic number is not 0x2e736e64 (four ASCII characters ".snd")
     * @returns         void
     */
    void read_file();

    /**
     * @brief           Apply the process algorithm corresponding to the value of the processing_algorithm class variable.
     * @details         To change the algorithm to use use the setter class method set_processing_algorithm.
     *
     * @returns         void
     */
    void apply_processing_algorithm();

    /**
     * @brief           Parse the file style, path and computed features values in a ready to use csv line.
     * @details         The format of the csv file correspond to the one described in get_csv_line_header and the computed features values depend on the processing algorithm used.
     *
     * @returns         the created line
     */
    std::string get_csv_line();

    /**
     * @brief           Return a ready to use csv header for the .au file features.
     * @details         The format of the csv file depend on the processing algorithm used;
     *                  When using the stft algorithm, the header format is the following: <BIN_AVG0,...,BIN_AVG255,BIN_STDEV0,...,BIN_STDEV254,style,fileName>. BIN_AVG and BIN_STDEV are casted to string from real_t type with BIN_AVG corresponding to the average values and BIN_STDEV to the standard deviation values.
     *                  When using the mfcc algorithm, the header format is the following: <SIGNALENERGY_AVG,BIN_AVG0,...,BIN_AVG18,SIGNALENERGY_STDEV,BIN_STDEV0,...,BIN_STDEV18,style,fileName>. SIGNALENERGY_AVG, BIN_AVG, SIGNALENERGY_STDEV and BIN_STDEV are casted to string from real_t type with BIN_AVG corresponding to the average values, BIN_STDEV to the standard deviation values and SIGNALENERGY_AVG and SIGNALENERGY_STDEV to the avg or stdev signal energy.
     *                  Style and FileName are string with the double quote.
     * @param[in]       processing_algorithm a AuFileProcessingAlgorithm enum variable that represent the process algorithm to use (default value: AuFileProcessor::DEFAULT_PROCESSING_ALGORITHM)
     *
     * @returns         the created csv header
     */
    static std::string get_csv_line_header(AuFileProcessingAlgorithm processing_algorithm = AuFileProcessor::DEFAULT_PROCESSING_ALGORITHM);

private:
    /**
     * Variable that store the .au file path.
     */
    std::filesystem::path file_path;
    /**
     * Constant that store the .au file default process algorithm (STFT)
     */
    static const AuFileProcessingAlgorithm DEFAULT_PROCESSING_ALGORITHM = AuFileProcessingAlgorithm::STFT;
    /**
     * Variable that store the .au file current process algorithm.
     */
    AuFileProcessingAlgorithm processing_algorithm;
    /**
     * Variable that store the .au file music style.
     * This value is obtained through the parsing of the file name and is done the object Constructor.
     */
    std::string music_style;

    /**
     * Variable that store the .au file magic number.
     * This value is obtained through the parsing of the file header and is done in the read_file class method.
     * Should be 0x2e736e64 (four ASCII characters ".snd").
     */
    word_t magic_number;
    /**
     * Variable that store the .au file data offset.
     * This value is obtained through the parsing of the file header and is done in the read_file class method.
     * It must be divisible by 8.
     * The minimum valid number is 24, since this is the header length (six 32-bit words) with no space reserved for extra information (the annotation field).
     * The minimum valid number with an annotation field present is 32 (decimal).
     */
    word_t data_offset;
    /**
     * Variable that store the .au file data size.
     * This value is obtained through the parsing of the file header and is done in the read_file class method.
     * If the value is 0xffffffff (4294967295), it means that the data ize is unknown.
     */
    word_t data_size;
    /**
     * Variable that store the .au file encoding format.
     * This value is obtained through the parsing of the file header and is done in the read_file class method.
     * It must be a value available in the AuFileEncodingFormat enum or the file will not be processable.
     */
    word_t encoding;
    /**
     * Variable that store the .au file sample rate (in sample/sec).
     * This value is obtained through the parsing of the file header and is done in the read_file class method.
     */

    word_t sample_rate;
    /**
     * Variable that store the .au file number of interleaved channels.
     * This value is obtained through the parsing of the file header and is done in the read_file class method.
     * Example of valid channels: 1 for mono, 2 for stereo, more channels possible, but may not be supported by all readers.
     */
    word_t channels;

    /**
     * Variable that store the .au file raw data in a vector.
     * This vector is obtained through the parsing of the file data and is done in the read_file class method.
     */
    real_vector_t raw_data;
    /**
     * Variable that store the.au file features average in a vector.
     * This is a vector of size SIZE_FFT(for STFT) or CEPSTRAL_COEFF(for MFCC) that contain the average of all frequencies after the stft/mfcc have been computed.
     */
    real_vector_t features_average{};
    /**
     * Variable that store the .au file features standard deviation in a vector.
     * This is a vector of size SIZE_FFT(for STFT) or CEPSTRAL_COEFF(for MFCC) that contain the standard deviation of all frequencies after the stft/mfcc have been computed.
     */
    real_vector_t features_standard_deviation{};

    /**
     * @brief           Read next word from the file ifstream.
     * @details         A word (type word_t) is a 32 bit unsigned integer.
     *                  The file is read byte per byte and each byte is converted from little endian to big endian.
     * @param[in]       file a std::ifstream object of the .au file to read.
     *
     * @returns         the extracted word
     */
    word_t get_next_word(std::ifstream &file);

    /**
     * @brief           Read next data from the file ifstream.
     * @details         A data (type int16_t) is a 16 bit signed integer.
     *                  The file is read byte per byte and each byte is converted from little endian to big endian.
     * @param[in]       file a std::ifstream object of the .au file to read.
     *
     * @returns         the extracted data
     */
    int16_t get_next_data(std::ifstream &file);

    /**
     * @brief           Read all music data from the file ifstream.
     * @details         The data is read from the data offset to the end of the file using the get_next_data class method and is stored in the raw_data class variable.
     *                  The available encoding format are listed in the AuFileEncodingFormat enumeration.
     * @param[in]       file a std::ifstream object of the .au file to read.
     *
     * @trhow           <std::domain_error("Unsupported data encoding!")> Throw an exception if the encoding is not supported.
     * @returns         void
     */
    void read_raw_data(std::ifstream &file);

    /**
     * @brief           Apply the stft to the raw data and save the average and standard deviation to their corresponding class variables.
     * @details         The stft extract from the raw data 2 list of data block, one with a step size of N and one with a step size of N/2.
     *                  Then a hamming window is applied using the windowing helper function and the representation in the frequency domain is computed using the ite_dit_fft (Iterative Direct Transform Fourrier) helper function.
     *                  Finally, the real value of only half of v1 and v2 is kept because the fft result is symmetrical regarding the origin and the average and standard deviation of each frequency are computed.
     *
     * @returns         void
     */
    void apply_stft();

    /**
     * @brief           Apply the mfcc to the raw data and save the average and standard deviation to their corresponding class variables.
     * @details         The mfcc, like the stft, extract from the raw data 2 list of data block, one with a step size of N and one with a step size of N/2.
     *                  After the computation of a list of filter depending on the min and max frequency and the number of filters wanted, a hamming window is applied using the windowing helper function.
     *                  The log value of this data block signal energy is then computed.
     *                  Then the Iterative Direct Transform Fourrier is applied, the results is converted from complex to real and only half of v1 and v2 is kept because the fft result is symmetrical regarding the origin.
     *                  The filters are then applied using the apply_filterbank helper function followed by the the Discret Cosinus Transform using the dct2 helper function
     *                  Finally, the average and standard deviation of all values are computed.
     *
     * @returns          void
     */
    void apply_mfcc();

    /**
     * @brief           Normalize the features_average and features_standard_deviation vectors using the formula: normalized_vector = (vector - mean) / stdev
     *
     *  @returns        void
     */
     void normalize_features();
};

#endif //AU_FILE_PROCESSOR
