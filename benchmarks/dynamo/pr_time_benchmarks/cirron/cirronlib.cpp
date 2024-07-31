#include <stdio.h>
#include <string.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <errno.h>
#include <sys/ioctl.h>
#include <stdint.h>
#include <stdlib.h>

#if defined(__linux__)
#include <linux/perf_event.h>
#elif defined(__APPLE__)
#include "apple_arm_events.h"
#endif

struct counter
{
    uint64_t time_enabled_ns;
    uint64_t instruction_count;
    uint64_t branch_misses;
    uint64_t page_faults;
};

extern "C"
{
    int start();
    int end(int fd, struct counter *out);
}

#if defined(__linux__)
struct read_format
{
    uint64_t nr;
    uint64_t time_enabled;
    uint64_t time_running;
    struct
    {
        uint64_t value;
    } values[];
};

struct perf_event_config
{
    uint64_t type;
    uint64_t config;
};

struct perf_event_config events[] = {
    {PERF_TYPE_HARDWARE, PERF_COUNT_HW_INSTRUCTIONS},
    {PERF_TYPE_HARDWARE, PERF_COUNT_HW_BRANCH_MISSES},
    {PERF_TYPE_SOFTWARE, PERF_COUNT_SW_PAGE_FAULTS},
    // {PERF_TYPE_SOFTWARE, PERF_COUNT_SW_CONTEXT_SWITCHES}, For whatever reason, these two always show 0
    // {PERF_TYPE_SOFTWARE, PERF_COUNT_SW_CPU_MIGRATIONS},
};

const int NUM_EVENTS = sizeof(events) / sizeof(events[0]);
#elif defined(__APPLE__)
u64 counters_0[KPC_MAX_COUNTERS] = {0};
usize counter_map[KPC_MAX_COUNTERS] = {0};
#endif

int start()
{
#if defined(__linux__)
    // Construct base perf_event_attr struct
    struct perf_event_attr attr;
    memset(&attr, 0, sizeof(attr));
    attr.size = sizeof(attr);
    attr.disabled = 1;
    attr.exclude_kernel = 1;
    attr.exclude_hv = 1;
    attr.sample_period = 0;
    attr.read_format = PERF_FORMAT_GROUP | PERF_FORMAT_TOTAL_TIME_ENABLED | PERF_FORMAT_TOTAL_TIME_RUNNING;

    int group = -1;
    int leader_fd;

    // Enable every event in perf_event_config
    for (int i = 0; i < NUM_EVENTS; i++)
    {
        attr.type = events[i].type;
        attr.config = events[i].config;

        int fd = syscall(SYS_perf_event_open, &attr, 0, -1, group, 0);
        if (fd == -1)
        {
            fprintf(stderr, "Failed to open event %lu: %s.\n", events[i].config, strerror(errno));
            return -1;
        }

        if (i == 0)
        {
            group = fd;
            leader_fd = fd;
        }
    }

    // Enable the event group
    if (ioctl(leader_fd, PERF_EVENT_IOC_ENABLE, PERF_IOC_FLAG_GROUP) == -1)
    {
        fprintf(stderr, "Failed to enable perf events: %s.\n", strerror(errno));
        // Consider cleaning up previously opened file descriptors here
        return -1;
    }

    return leader_fd;
#elif defined(__APPLE__)
    // load dylib
    if (!lib_init())
    {
        printf("Error: %s\n", lib_err_msg);
        return 1;
    }

    // check permission
    int force_ctrs = 0;
    if (kpc_force_all_ctrs_get(&force_ctrs))
    {
        printf("Permission denied, xnu/kpc requires root privileges.\n");
        return 1;
    }

    // load pmc db
    int ret = 0;
    kpep_db *db = NULL;
    if ((ret = kpep_db_create(NULL, &db)))
    {
        printf("Error: cannot load pmc database: %d.\n", ret);
        return 1;
    }

    // create a config
    kpep_config *cfg = NULL;
    if ((ret = kpep_config_create(db, &cfg)))
    {
        printf("Failed to create kpep config: %d (%s).\n",
               ret, kpep_config_error_desc(ret));
        return 1;
    }
    if ((ret = kpep_config_force_counters(cfg)))
    {
        printf("Failed to force counters: %d (%s).\n",
               ret, kpep_config_error_desc(ret));
        return 1;
    }

    // get events
    const usize ev_count = sizeof(profile_events) / sizeof(profile_events[0]);
    kpep_event *ev_arr[ev_count] = {0};
    for (usize i = 0; i < ev_count; i++)
    {
        const event_alias *alias = profile_events + i;
        ev_arr[i] = get_event(db, alias);
        if (!ev_arr[i])
        {
            printf("Cannot find event: %s.\n", alias->alias);
            return 1;
        }
    }

    // add event to config
    for (usize i = 0; i < ev_count; i++)
    {
        kpep_event *ev = ev_arr[i];
        if ((ret = kpep_config_add_event(cfg, &ev, 0, NULL)))
        {
            printf("Failed to add event: %d (%s).\n",
                   ret, kpep_config_error_desc(ret));
            return 1;
        }
    }

    // prepare buffer and config
    u32 classes = 0;
    usize reg_count = 0;
    kpc_config_t regs[KPC_MAX_COUNTERS] = {0};
    if ((ret = kpep_config_kpc_classes(cfg, &classes)))
    {
        printf("Failed get kpc classes: %d (%s).\n",
               ret, kpep_config_error_desc(ret));
        return 1;
    }
    if ((ret = kpep_config_kpc_count(cfg, &reg_count)))
    {
        printf("Failed get kpc count: %d (%s).\n",
               ret, kpep_config_error_desc(ret));
        return 1;
    }
    if ((ret = kpep_config_kpc_map(cfg, counter_map, sizeof(counter_map))))
    {
        printf("Failed get kpc map: %d (%s).\n",
               ret, kpep_config_error_desc(ret));
        return 1;
    }
    if ((ret = kpep_config_kpc(cfg, regs, sizeof(regs))))
    {
        printf("Failed get kpc registers: %d (%s).\n",
               ret, kpep_config_error_desc(ret));
        return 1;
    }

    // set config to kernel
    if ((ret = kpc_force_all_ctrs_set(1)))
    {
        printf("Failed force all ctrs: %d.\n", ret);
        return 1;
    }
    if ((classes & KPC_CLASS_CONFIGURABLE_MASK) && reg_count)
    {
        if ((ret = kpc_set_config(classes, regs)))
        {
            printf("Failed set kpc config: %d.\n", ret);
            return 1;
        }
    }

    // start counting
    if ((ret = kpc_set_counting(classes)))
    {
        printf("Failed set counting: %d.\n", ret);
        return 1;
    }
    if ((ret = kpc_set_thread_counting(classes)))
    {
        printf("Failed set thread counting: %d.\n", ret);
        return 1;
    }

    // get counters before
    if ((ret = kpc_get_thread_counters(0, KPC_MAX_COUNTERS, counters_0)))
    {
        printf("Failed get thread counters before: %d.\n", ret);
        return 1;
    }

    return 0;
#else
    printf("This systems seems to be neither Linux, nor ARM OSX, so I don't know how to proceeed.\nIf this is a mistake, please open an issue on the GitHub repository.\n");
    return -1;
#endif
}

int end(int fd, struct counter *out)
{
#if defined(__linux__)
    if (out == NULL)
    {
        fprintf(stderr, "Error: 'out' pointer is NULL in end().\n");
        return -1;
    }

    // Disable the event group
    if (ioctl(fd, PERF_EVENT_IOC_DISABLE, PERF_IOC_FLAG_GROUP) == -1)
    {
        fprintf(stderr, "Error disabling perf event (fd: %d): %s\n", fd, strerror(errno));
        return -1;
    }

    // Allocate buffer for reading results
    int size = sizeof(struct read_format) + (sizeof(uint64_t) * NUM_EVENTS);
    struct read_format *buffer = (struct read_format *)malloc(size);
    if (!buffer)
    {
        fprintf(stderr, "Failed to allocate memory for read buffer.\n");
        return -1;
    }

    // Read results
    int ret_val = read(fd, buffer, size);
    if (ret_val == -1)
    {
        fprintf(stderr, "Error reading perf event results: %s\n", strerror(errno));
        free(buffer);
        return -1;
    }
    else if (ret_val != size)
    {
        fprintf(stderr, "Error reading perf event results: read %d bytes, expected %d\n", ret_val, size);
        free(buffer);
        return -1;
    }

    // Assign time_enabled_ns
    out->time_enabled_ns = buffer->time_enabled;

    // Directly assign values to struct fields treating them as an array 8)
    uint64_t *counter_ptr = (uint64_t *)out;
    counter_ptr++; // Now points to instruction_count, the first counter field

    for (int i = 0; i < NUM_EVENTS; i++)
    {
        counter_ptr[i] = buffer->values[i].value;
    }

    close(fd);
    free(buffer);
    return 0;
#elif defined(__APPLE__)
    // get counters after
    int ret = 0;
    u64 counters_1[KPC_MAX_COUNTERS] = {0};
    if ((ret = kpc_get_thread_counters(0, KPC_MAX_COUNTERS, counters_1)))
    {
        printf("Failed get thread counters after: %d.\n", ret);
        return 1;
    }

    kpc_set_counting(0);
    kpc_set_thread_counting(0);
    kpc_force_all_ctrs_set(0);

    out->time_enabled_ns = 0;
    out->instruction_count = counters_1[counter_map[1]] - counters_0[counter_map[1]];
    out->page_faults = 0;
    out->branch_misses = counters_1[counter_map[3]] - counters_0[counter_map[3]];
    return 0;
#else
    printf("This systems seems to be neither Linux, nor OSX, so I don't know how to proceeed.\nIf this is a mistake, please open an issue on the GitHub repository.\n");
    return -1;
#endif
}