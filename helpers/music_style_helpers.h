#ifndef EML_MUSIC_STYLE_HELPERS_H
#define EML_MUSIC_STYLE_HELPERS_H

#include <string>
#include <stdexcept>

enum class MUSIC_STYLE {
    BLUES = 0,
    CLASSICAL = 1,
    COUNTRY = 2,
    DISCO = 3,
    HIPHOP = 4,
    JAZZ = 5,
    METAL = 6,
    POP = 7,
    REGGAE = 8,
    ROCK = 9
};

static inline std::string music_style_to_string(const MUSIC_STYLE &s) {
    switch (s) {
        case MUSIC_STYLE::BLUES: {
            return "blues";
        }
        case MUSIC_STYLE::CLASSICAL: {
            return "classical";
        }
        case MUSIC_STYLE::COUNTRY: {
            return "country";
        }
        case MUSIC_STYLE::DISCO: {
            return "disco";
        }
        case MUSIC_STYLE::HIPHOP: {
            return "hiphop";
        }
        case MUSIC_STYLE::JAZZ: {
            return "jazz";
        }
        case MUSIC_STYLE::METAL: {
            return "metal";
        }
        case MUSIC_STYLE::POP: {
            return "pop";
        }
        case MUSIC_STYLE::REGGAE: {
            return "reggae";
        }
        case MUSIC_STYLE::ROCK: {
            return "rock";
        }
        default: {
            throw std::logic_error("to_string: MUSIC_STYLE enum values not found.");
        }
    }
}

static inline MUSIC_STYLE music_style_from_string(std::string str) {
    if (str == "blues") {
        return MUSIC_STYLE::BLUES;
    } else if (str == "classical") {
        return MUSIC_STYLE::CLASSICAL;
    } else if (str == "country") {
        return MUSIC_STYLE::COUNTRY;
    } else if (str == "disco") {
        return MUSIC_STYLE::DISCO;
    } else if (str == "hiphop") {
        return MUSIC_STYLE::HIPHOP;
    } else if (str == "jazz") {
        return MUSIC_STYLE::JAZZ;
    } else if (str == "metal") {
        return MUSIC_STYLE::METAL;
    } else if (str == "pop") {
        return MUSIC_STYLE::POP;
    } else if (str == "reggae") {
        return MUSIC_STYLE::REGGAE;
    } else if (str == "rock") {
        return MUSIC_STYLE::ROCK;
    } else {
        throw std::logic_error("from_string: MUSIC_STYLE enum values not found.");
    }
}

static inline MUSIC_STYLE music_style_from_int(unsigned int s) {
    switch (s) {
        case 0: {
            return MUSIC_STYLE::BLUES;
        }
        case 1: {
            return MUSIC_STYLE::CLASSICAL;
        }
        case 2: {
            return MUSIC_STYLE::COUNTRY;
        }
        case 3: {
            return MUSIC_STYLE::DISCO;
        }
        case 4: {
            return MUSIC_STYLE::HIPHOP;
        }
        case 5: {
            return MUSIC_STYLE::JAZZ;
        }
        case 6: {
            return MUSIC_STYLE::METAL;
        }
        case 7: {
            return MUSIC_STYLE::POP;
        }
        case 8: {
            return MUSIC_STYLE::REGGAE;
        }
        case 9: {
            return MUSIC_STYLE::ROCK;
        }
        default: {
            throw std::logic_error("to_int: MUSIC_STYLE enum values not found.");
        }
    }
}

std::ostream &operator<<(std::ostream &os, MUSIC_STYLE music_style) {
    os << music_style_to_string(music_style);
    return os;
}


#endif //EML_MUSIC_STYLE_HELPERS_H
