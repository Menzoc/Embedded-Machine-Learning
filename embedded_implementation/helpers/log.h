#include <iostream>

#ifdef NO_LOG
#define LOG(type) LOGGER()
#else
#define LOG(type) LOGGER(type, __FILE__, __LINE__, __FUNCTION__)
#endif /* NO_LOG */

#ifndef _DEBUG_LOG_H
#define _DEBUG_LOG_H

/**
 * @brief List of log levels  
 */
enum log_level_enum {
    LOG_DEBUG,      /**< Use for debuging messages */
    LOG_INFO,       /**< Use for basic execution message */
    LOG_WARNING,    /**< Use for non blocking errors */
    LOG_ERROR       /**< Use for critical errors */
};

/**
 * @brief Hold structure for log config
 */
struct log_struct {
    bool header = true;                 /**< True to print a header before the log message; False else */
    log_level_enum level = LOG_DEBUG;   /**< Default Log level */
    bool keep_full_path = false;           /**< True to keep full path from __FILE__; False to only keep filename and extension */
};

extern log_struct LOGGING_CONFIG;

class LOGGER {
public:
    LOGGER() {
        no_log = true;
    }

    LOGGER(log_level_enum level, const std::string &file, int line, const std::string &function) {
        no_log = false;
        log_level = level;

        if (LOGGING_CONFIG.header) {
            if (LOGGING_CONFIG.keep_full_path) {
                operator<<("[" + getLabel(level) + "] " + file + ":" + std::to_string(line) + ":" + function + "()\t");
            } else {
                operator<<("[" + getLabel(level) + "] " + file.substr(file.find_last_of("/\\") + 1) + ":" + std::to_string(line) + ":" + function + "()\t");
            }
        }
    }

    ~LOGGER() {
        if (log_stream_opened) {
            std::cout << std::endl;
        }
        log_stream_opened = false;
    }

    template<class T>
    LOGGER &operator<<(const T &msg) {
        if (log_level >= LOGGING_CONFIG.level && !no_log) {
            std::cout << msg;
            log_stream_opened = true;
        }
        return *this;
    }

private:
    bool log_stream_opened = false;
    bool no_log = false;
    log_level_enum log_level = LOG_DEBUG;

    inline std::string getLabel(log_level_enum level) {
        std::string log_level_label;
        switch (level) {
            case LOG_DEBUG: {
                log_level_label = "DEBUG";
                break;
            }
            case LOG_INFO: {
                log_level_label = "INFO";
                break;
            }
            case LOG_WARNING: {
                log_level_label = "WARNING";
                break;
            }
            case LOG_ERROR: {
                log_level_label = "ERROR";
                break;
            }
        }
        return log_level_label;
    }
};

#endif /* _DEBUG_LOG_H */