#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cstdint>
#include <cstdio>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <psapi.h>
#elif defined(__APPLE__)
#include <mach/mach.h>
#elif defined(__linux__)
#include <fstream>
#include <limits>
#include <string>
#endif

// A large upper bound representative of TTS schedulers.
static constexpr size_t k_graph_capacity = 256*1024;

struct memory_snapshot {
    uint64_t primary_bytes;
    uint64_t resident_bytes;
    const char * primary_label;
};

#if defined(__linux__)
static uint64_t linux_status_bytes(const char * wanted_key) {
    std::ifstream status("/proc/self/status");
    std::string key;

    while (status >> key) {
        if (key == wanted_key) {
            uint64_t kib = 0;
            status >> kib;
            return kib*1024;
        }
        status.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    }
    return 0;
}
#endif

static memory_snapshot get_memory() {
#if defined(_WIN32)
    PROCESS_MEMORY_COUNTERS_EX counters = {};
    counters.cb = sizeof(counters);
    GetProcessMemoryInfo(
        GetCurrentProcess(),
        reinterpret_cast<PROCESS_MEMORY_COUNTERS *>(&counters),
        sizeof(counters));
    return { counters.PrivateUsage, counters.WorkingSetSize, "committed/private" };
#elif defined(__APPLE__)
    mach_task_basic_info info = {};
    mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
    task_info(
        mach_task_self(),
        MACH_TASK_BASIC_INFO,
        reinterpret_cast<task_info_t>(&info),
        &count);
    return { info.virtual_size, info.resident_size, "virtual" };
#elif defined(__linux__)
    return { linux_status_bytes("VmSize:"), linux_status_bytes("VmRSS:"), "virtual" };
#else
    return { 0, 0, "memory" };
#endif
}

static void print_memory(const char * label, const memory_snapshot & memory) {
    constexpr double bytes_per_mib = 1024.0*1024.0;
    std::printf(
        "%s: %s %.2f MiB, resident %.2f MiB\n",
        label,
        memory.primary_label,
        memory.primary_bytes/bytes_per_mib,
        memory.resident_bytes/bytes_per_mib);
}

int main() {
    ggml_backend_t backend = ggml_backend_cpu_init();
    ggml_backend_t backends[] = { backend };

    const memory_snapshot before = get_memory();
    ggml_backend_sched_t sched = ggml_backend_sched_new(
        backends, nullptr, 1, k_graph_capacity, false, true);
    const memory_snapshot after = get_memory();

    print_memory("before scheduler memory", before);
    print_memory("after scheduler memory", after);

    constexpr double bytes_per_mib = 1024.0*1024.0;
    std::printf(
        "scheduler memory delta: %s %+.2f MiB, resident %+.2f MiB\n",
        after.primary_label,
        (static_cast<double>(after.primary_bytes) - static_cast<double>(before.primary_bytes))/bytes_per_mib,
        (static_cast<double>(after.resident_bytes) - static_cast<double>(before.resident_bytes))/bytes_per_mib);

    ggml_backend_sched_free(sched);
    ggml_backend_free(backend);
    return 0;
}
