cmake_minimum_required(VERSION 3.1)
project(VideoVisualization)

# Set C++17 standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenCV
find_package(OpenCV REQUIRED)
# Find Pangolin
find_package(Pangolin REQUIRED)
find_package(g2o REQUIRED)
Find_Package(Sophus REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

include_directories(${G2O_INCLUDE_DIRS})
link_directories(${G2O_LIBRARY_DIRS})



# List of g2o libraries
set(G2O_LIBS
    g2o_solver_cholmod    
    g2o_core
    g2o_stuff
    g2o_types_slam3d
    # Add additional g2o libraries as needed
)

# Create the executable
add_executable(video_visualization main.cpp)

# Link libraries
# target_link_libraries(video_visualization ${OpenCV_LIBS} ${Pangolin_LIBRARIES} ${G2O_LIBS})
find_package(OpenMP)

target_link_libraries(video_visualization 
${OpenCV_LIBS} 
${Pangolin_LIBRARIES} 
${G2O_LIBS} 
${CSPARSE_LIBRARY} # CSPARSE_LIBRARY는 실제 찾은 CSparse 라이브러리의 변수명과 일치해야 합니다.

g2o_solver_cholmod
g2o_core 
g2o_stuff 
g2o_types_sba 
g2o_types_slam3d 
OpenMP::OpenMP_CXX
)
