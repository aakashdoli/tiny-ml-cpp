#pragma once

#include <atomic>
#include <cstddef>
#include <new>

namespace tinyml::baseband {

/**
 * @brief Lock-free Single-Producer Single-Consumer (SPSC) Circular Ring Buffer.
 * 
 * Crucial for multi-core DSP architectures in baseband processing:
 * - Transfers high-speed radio IQ sample streams between pipeline stages.
 * - Zero dynamic heap allocation.
 * - Cache-line aligned (64 bytes) to eliminate false sharing between cores.
 * - Wait-free microsecond read/write operations without mutexes or kernel locks.
 */
template <typename T, std::size_t Capacity>
class SPSCRingBuffer {
    static_assert((Capacity & (Capacity - 1)) == 0, "Capacity must be a power of two for bitwise wrap-around");
    static_assert(Capacity >= 2, "Capacity must be at least 2");

public:
    constexpr SPSCRingBuffer() noexcept : head_(0), tail_(0) {}

    ~SPSCRingBuffer() = default;
    SPSCRingBuffer(const SPSCRingBuffer&) = delete;
    SPSCRingBuffer& operator=(const SPSCRingBuffer&) = delete;

    /**
     * @brief Pushes an item into the buffer (Producer Core only).
     * @return true if successfully pushed, false if buffer is full.
     */
    bool push(const T& item) noexcept {
        const std::size_t current_head = head_.load(std::memory_order_relaxed);
        const std::size_t current_tail = tail_.load(std::memory_order_acquire);

        if ((current_head - current_tail) >= Capacity) {
            return false; // Buffer full: prevent overflow without dropping state
        }

        buffer_[current_head & (Capacity - 1)] = item;
        head_.store(current_head + 1, std::memory_order_release);
        return true;
    }

    /**
     * @brief Pops an item from the buffer (Consumer Core only).
     * @return true if successfully popped, false if buffer is empty.
     */
    bool pop(T& item) noexcept {
        const std::size_t current_tail = tail_.load(std::memory_order_relaxed);
        const std::size_t current_head = head_.load(std::memory_order_acquire);

        if (current_head == current_tail) {
            return false; // Buffer empty: consumer underflow
        }

        item = buffer_[current_tail & (Capacity - 1)];
        tail_.store(current_tail + 1, std::memory_order_release);
        return true;
    }

    std::size_t size() const noexcept {
        const std::size_t h = head_.load(std::memory_order_relaxed);
        const std::size_t t = tail_.load(std::memory_order_relaxed);
        return h >= t ? (h - t) : 0;
    }

    bool empty() const noexcept {
        return head_.load(std::memory_order_relaxed) == tail_.load(std::memory_order_relaxed);
    }

    constexpr std::size_t capacity() const noexcept {
        return Capacity;
    }

private:
    T buffer_[Capacity];

    // Hardware cache-line alignment prevents false sharing between producer and consumer cores
    alignas(64) std::atomic<std::size_t> head_{0};
    alignas(64) std::atomic<std::size_t> tail_{0};
};

}  // namespace tinyml::baseband
