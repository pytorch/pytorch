/*
    Copyright (c) 2005-2018 Intel Corporation

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.




*/

#if __APPLE__

#define HARNESS_CUSTOM_MAIN 1
#include "harness.h"
#include <cstdlib>
#include "tbb/task_scheduler_init.h"

#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include <signal.h>
#include <errno.h>

bool exec_test(const char *self) {
    int status = 1;
    pid_t p = fork();
    if(p < 0) {
        REPORT("fork error: errno=%d: %s\n", errno, strerror(errno));
        return true;
    }
    else if(p) { // parent
        if(waitpid(p, &status, 0) != p) {
            REPORT("wait error: errno=%d: %s\n", errno, strerror(errno));
            return true;
        }
        if(WIFEXITED(status)) {
            if(!WEXITSTATUS(status)) return false; // ok
            else REPORT("child has exited with return code 0x%x\n", WEXITSTATUS(status));
        } else {
            REPORT("child error 0x%x:%s%s ", status, WIFSIGNALED(status)?" signalled":"",
                WIFSTOPPED(status)?" stopped":"");
            if(WIFSIGNALED(status))
                REPORT("%s%s", sys_siglist[WTERMSIG(status)], WCOREDUMP(status)?" core dumped":"");
            if(WIFSTOPPED(status))
                REPORT("with %d stop-code", WSTOPSIG(status));
            REPORT("\n");
        }
    }
    else { // child
        // reproduces error much often
        execl(self, self, "0", NULL);
        REPORT("exec fails %s: %d: %s\n", self, errno, strerror(errno));
        exit(2);
    }
    return true;
}

HARNESS_EXPORT
int main( int argc, char * argv[] ) {
    MinThread = 3000;
    ParseCommandLine( argc, argv );
    if( MinThread <= 0 ) {
        tbb::task_scheduler_init init( 2 ); // even number required for an error
    } else {
        for(int i = 0; i<MinThread; i++) {
            if(exec_test(argv[0])) {
                REPORT("ERROR: execution fails at %d-th iteration!\n", i);
                exit(1);
            }
        }
        REPORT("done\n");
    }
}

#else /* !__APPLE__ */

#define HARNESS_NO_PARSE_COMMAND_LINE 1
#include "harness.h"

int TestMain () {
    return Harness::Skipped;
}

#endif /* !__APPLE__ */
