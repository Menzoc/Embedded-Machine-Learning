#ifndef PRINT_HELPERS_H
#define PRINT_HELPERS_H

#include <iterator>
#include "globals.h"

template<typename T>
std::ostream &operator<<(std::ostream &os, const std::vector<T> &v) {
    os << "[";
    for (typename std::vector<T>::const_iterator i = v.begin(); i != v.end(); ++i) {
        if (i == v.end() - 1) {
            os << *i;
        } else {
            os << *i << ", ";
        }
    }
    os << "]";
    return os;
}

template<typename T, std::size_t SIZE>
std::ostream &operator<<(std::ostream &os, const std::array<T, SIZE> &a) {
    os << "(";
    for (typename std::array<T, SIZE>::const_iterator i = a.begin(); i != a.end(); ++i) {
        if (i == a.end() - 1) {
            os << *i;
        } else {
            os << *i << ", ";
        }
    }
    os << ")";
    return os;
}


#endif //PRINT_HELPERS_H
