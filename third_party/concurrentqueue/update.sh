#!/bin/bash

# Create the moodycamel directory if it doesn't exist
mkdir -p moodycamel

# Download the concurrentqueue.h file
curl -o moodycamel/concurrentqueue.h https://raw.githubusercontent.com/cameron314/concurrentqueue/master/concurrentqueue.h

# Download the lightweightsemaphore.h file
curl -o moodycamel/lightweightsemaphore.h https://raw.githubusercontent.com/cameron314/concurrentqueue/master/lightweightsemaphore.h

# Download the LICENSE.md file
curl -o moodycamel/LICENSE.md https://raw.githubusercontent.com/cameron314/concurrentqueue/master/LICENSE.md
