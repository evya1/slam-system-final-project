cmake_minimum_required(VERSION 3.16)
project(slam_system)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgproc features2d calib3d highgui imgcodecs)
find_package(Pangolin REQUIRED)

add_executable(slam_system main.cpp)

target_compile_definitions(slam_system PRIVATE PROJECT_ROOT_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}\")

target_include_directories(slam_system PRIVATE
        ${EIGEN3_INCLUDE_DIR}
        ${OpenCV_INCLUDE_DIRS}
        ${Pangolin_INCLUDE_DIRS}
)

target_link_libraries(slam_system PRIVATE
        Eigen3::Eigen
        ${OpenCV_LIBS}
        ${Pangolin_LIBRARIES}
)