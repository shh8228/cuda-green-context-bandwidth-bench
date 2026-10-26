#pragma once

#include <cerrno>
#include <climits>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

struct BenchOptions {
    size_t buffer_mb = 512;
    int iterations = 20, gpu_id = 0, sm_step = 0, trials = 5, load_mode = 0;
    int blocks_per_sm = 32, tma_stages = 2, sm_count = 0, seed = 8228;
    bool help = false, external_single = false, external_probe = false;
    bool validate_only = false, skip_validation = false, require_dram = false;
    bool tma_copy_only = false;
    int tc_n = 8;
    std::string tc_consumer = "mma";
    std::string tc_schedule = "overlap";
};

inline unsigned long long parse_integer(const char *value, const char *name,
                                        unsigned long long minimum,
                                        unsigned long long maximum = INT_MAX)
{
    char *end = nullptr;
    errno = 0;
    unsigned long long number = std::strtoull(value, &end, 10);
    if (*value == '-' || *value == '\0' || errno || *end ||
        number < minimum || number > maximum)
        throw std::invalid_argument(std::string(name) + " is outside its valid integer range");
    return number;
}

inline BenchOptions parse_options(int argc, char **argv)
{
    BenchOptions o;
    bool tc_option = false;
    std::vector<const char *> positional;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") o.help = true;
        else if (arg == "--external-single") o.external_single = true;
        else if (arg == "--external-probe") o.external_probe = true;
        else if (arg == "--validate-only") o.validate_only = true;
        else if (arg == "--skip-validation") o.skip_validation = true;
        else if (arg == "--require-dram") o.require_dram = true;
        else if (arg == "--tma-copy-only") o.tma_copy_only = true;
        else if (arg == "--tc-consumer" || arg == "--tc-schedule") {
            if (++i == argc) throw std::invalid_argument(arg + " requires a value");
            if (arg == "--tc-consumer") o.tc_consumer = argv[i];
            else o.tc_schedule = argv[i];
            tc_option = true;
        }
        else if (arg == "--blocks-per-sm" || arg == "--tma-stages" ||
                 arg == "--sm-count" || arg == "--seed" || arg == "--tc-n") {
            if (++i == argc) throw std::invalid_argument(arg + " requires a value");
            int n = static_cast<int>(parse_integer(argv[i], arg.c_str(), arg == "--seed" ? 0 : 1));
            if (arg == "--blocks-per-sm") o.blocks_per_sm = n;
            else if (arg == "--tma-stages") o.tma_stages = n;
            else if (arg == "--sm-count") o.sm_count = n;
            else if (arg == "--tc-n") { o.tc_n = n; tc_option = true; }
            else o.seed = n;
        } else if (arg.rfind("--", 0) == 0)
            throw std::invalid_argument("Unknown option: " + arg);
        else positional.push_back(argv[i]);
    }
    if (o.help) return o;
    if (o.tc_n != 8 && o.tc_n != 16 && o.tc_n != 32)
        throw std::invalid_argument("--tc-n must be 8, 16, or 32");
    if (o.tc_consumer != "mma" && o.tc_consumer != "sum" && o.tc_consumer != "copy")
        throw std::invalid_argument("--tc-consumer must be mma, sum, or copy");
    if (o.tc_schedule != "serial" && o.tc_schedule != "overlap")
        throw std::invalid_argument("--tc-schedule must be serial or overlap");
    if (o.tma_stages != 2 && o.tma_stages != 4 && o.tma_stages != 8)
        throw std::invalid_argument("--tma-stages must be 2, 4, or 8");
    if (o.blocks_per_sm > 256)
        throw std::invalid_argument("--blocks-per-sm must be between 1 and 256");
    if (int(o.external_single) + int(o.external_probe) + int(o.validate_only) > 1 ||
        (o.sm_count && (o.external_single || o.external_probe || o.validate_only)))
        throw std::invalid_argument("Select only one execution mode");
    if (o.external_probe) {
        if (tc_option) throw std::invalid_argument("--tc-n/--tc-consumer/--tc-schedule requires load_mode 6");
        if (positional.size() > 1) throw std::invalid_argument("--external-probe accepts only gpu_id");
        if (!positional.empty()) o.gpu_id = parse_integer(positional[0], "gpu_id", 0);
        return o;
    }
    if (positional.size() > (o.external_single ? 5U : 6U))
        throw std::invalid_argument("Too many positional arguments");
    const char *names[] = {"buffer_MB", "iterations", "gpu_id", "sm_step", "trials", "load_mode"};
    if (o.external_single) o.load_mode = 1;
    for (size_t i = 0; i < positional.size(); ++i) {
        size_t field = o.external_single && i >= 3 ? i + 1 : i;
        const auto minimum = (field == 0 || field == 1 || field == 4) ? 1U : 0U;
        const auto maximum = field == 0
            ? std::numeric_limits<size_t>::max() / (1024ULL * 1024ULL) : INT_MAX;
        auto n = parse_integer(positional[i], names[field], minimum, maximum);
        switch (field) {
        case 0: o.buffer_mb = n; break;
        case 1: o.iterations = n; break;
        case 2: o.gpu_id = n; break;
        case 3: o.sm_step = n; break;
        case 4: o.trials = n; break;
        case 5: o.load_mode = n; break;
        }
    }
    if (o.load_mode > 6) throw std::invalid_argument("load_mode must be between 0 and 6");
    if (tc_option && o.load_mode != 6)
        throw std::invalid_argument("--tc-n/--tc-consumer/--tc-schedule requires load_mode 6");
    if (o.tma_copy_only && o.load_mode != 4)
        throw std::invalid_argument("--tma-copy-only requires load_mode 4");
    return o;
}
