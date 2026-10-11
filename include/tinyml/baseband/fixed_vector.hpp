#pragma once

#include <cstddef>
#include <iterator>
#include <new>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace tinyml::baseband {

/**
 * @brief Zero-heap, stack-allocated vector with compile-time bounded capacity.
 * 
 * Designed for hard real-time baseband and DSP systems where dynamic memory 
 * allocation (malloc/free) is strictly forbidden due to non-deterministic latency.
 */
template <typename T, std::size_t Capacity>
class FixedVector {
public:
    static_assert(Capacity > 0, "Capacity must be greater than zero");

    using value_type = T;
    using size_type = std::size_t;
    using pointer = T*;
    using const_pointer = const T*;
    using reference = T&;
    using const_reference = const T&;
    using iterator = T*;
    using const_iterator = const T*;

    constexpr FixedVector() noexcept : size_(0) {}

    FixedVector(size_type count, const T& value) : size_(0) {
        if (count > Capacity) throw std::length_error("FixedVector capacity exceeded");
        for (size_type i = 0; i < count; ++i) {
            push_back(value);
        }
    }

    ~FixedVector() {
        clear();
    }

    FixedVector(const FixedVector& other) : size_(0) {
        for (size_type i = 0; i < other.size_; ++i) {
            push_back(other[i]);
        }
    }

    FixedVector& operator=(const FixedVector& other) {
        if (this != &other) {
            clear();
            for (size_type i = 0; i < other.size_; ++i) {
                push_back(other[i]);
            }
        }
        return *this;
    }

    FixedVector(FixedVector&& other) noexcept : size_(0) {
        for (size_type i = 0; i < other.size_; ++i) {
            push_back(std::move(other[i]));
        }
        other.clear();
    }

    FixedVector& operator=(FixedVector&& other) noexcept {
        if (this != &other) {
            clear();
            for (size_type i = 0; i < other.size_; ++i) {
                push_back(std::move(other[i]));
            }
            other.clear();
        }
        return *this;
    }

    constexpr size_type capacity() const noexcept { return Capacity; }
    constexpr size_type size() const noexcept { return size_; }
    constexpr bool empty() const noexcept { return size_ == 0; }
    constexpr bool full() const noexcept { return size_ == Capacity; }

    void clear() noexcept {
        for (size_type i = 0; i < size_; ++i) {
            data_ptr()[i].~T();
        }
        size_ = 0;
    }

    void push_back(const T& value) {
        if (full()) throw std::length_error("FixedVector push_back on full buffer");
        ::new (static_cast<void*>(data_ptr() + size_)) T(value);
        ++size_;
    }

    void push_back(T&& value) {
        if (full()) throw std::length_error("FixedVector push_back on full buffer");
        ::new (static_cast<void*>(data_ptr() + size_)) T(std::move(value));
        ++size_;
    }

    template <typename... Args>
    reference emplace_back(Args&&... args) {
        if (full()) throw std::length_error("FixedVector emplace_back on full buffer");
        T* ptr = ::new (static_cast<void*>(data_ptr() + size_)) T(std::forward<Args>(args)...);
        ++size_;
        return *ptr;
    }

    void pop_back() noexcept {
        if (size_ > 0) {
            --size_;
            data_ptr()[size_].~T();
        }
    }

    reference operator[](size_type idx) noexcept { return data_ptr()[idx]; }
    const_reference operator[](size_type idx) const noexcept { return data_ptr()[idx]; }

    reference front() noexcept { return data_ptr()[0]; }
    const_reference front() const noexcept { return data_ptr()[0]; }

    reference back() noexcept { return data_ptr()[size_ - 1]; }
    const_reference back() const noexcept { return data_ptr()[size_ - 1]; }

    pointer data() noexcept { return data_ptr(); }
    const_pointer data() const noexcept { return data_ptr(); }

    iterator begin() noexcept { return data_ptr(); }
    const_iterator begin() const noexcept { return data_ptr(); }
    iterator end() noexcept { return data_ptr() + size_; }
    const_iterator end() const noexcept { return data_ptr() + size_; }

private:
    alignas(alignof(T)) std::byte storage_[Capacity * sizeof(T)];
    size_type size_{0};

    pointer data_ptr() noexcept {
        return reinterpret_cast<pointer>(storage_);
    }

    const_pointer data_ptr() const noexcept {
        return reinterpret_cast<const_pointer>(storage_);
    }
};

}  // namespace tinyml::baseband
