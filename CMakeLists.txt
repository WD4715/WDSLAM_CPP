# cmake_minimum_required(VERSION 3.10)
# project(VideoVisualization)

# find_package(OpenCV REQUIRED)
# include_directories(${OpenCV_INCLUDE_DIRS})

# add_executable(video_visualization main.cpp)
# target_link_libraries(video_visualization ${OpenCV_LIBS})


cmake_minimum_required(VERSION 3.10)
project(VideoVisualization)

# Find OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Find Pangolin
find_package(Pangolin REQUIRED)

# Create the executable
add_executable(video_visualization main.cpp)

# Link libraries
target_link_libraries(video_visualization ${OpenCV_LIBS} ${Pangolin_LIBRARIES})

# Include directories
include_directories(${Pangolin_INCLUDE_DIRS})
