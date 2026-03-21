cmake_minimum_required(VERSION 3.16)
project(slam_system)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgproc features2d calib3d highgui imgcodecs)
find_package(Pangolin REQUIRED CONFIG)

add_executable(slam_system main.cpp)

target_link_libraries(slam_system PRIVATE
        Eigen3::Eigen
        ${OpenCV_LIBS}
)

if (TARGET Pangolin::pangolin)
    target_link_libraries(slam_system PRIVATE Pangolin::pangolin)
elseif (TARGET Pangolin::Pangolin)
    target_link_libraries(slam_system PRIVATE Pangolin::Pangolin)
elseif (TARGET pangolin)
    target_link_libraries(slam_system PRIVATE pangolin)
elseif (DEFINED Pangolin_LIBRARIES)
    target_link_libraries(slam_system PRIVATE ${Pangolin_LIBRARIES})
else ()
    message(FATAL_ERROR "Pangolin found but no known exported target / libs")
endif ()

if (DEFINED Pangolin_INCLUDE_DIRS)
    target_include_directories(slam_system PRIVATE ${Pangolin_INCLUDE_DIRS})
endif ()