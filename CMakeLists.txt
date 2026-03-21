cmake_minimum_required(VERSION 3.16)

project(slam_system VERSION 0.2.0 LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Build type" FORCE)
endif()

option(SLAM_SYSTEM_ENABLE_WARNINGS "Enable common compiler warnings" ON)

# ---------------------------------------------------------------------------
# Source files
# ---------------------------------------------------------------------------
set(SLAM_SOURCES
    main.cpp
    slam_map.cpp
    feature_matcher.cpp
    epipolar_geometry.cpp
    triangulation.cpp
    pnp_relocalizer.cpp
    diagnostics.cpp
    optimization.cpp
    frontend.cpp
    viewer.cpp
)

add_executable(slam_system ${SLAM_SOURCES})
target_compile_features(slam_system PRIVATE cxx_std_17)

if(SLAM_SYSTEM_ENABLE_WARNINGS)
    target_compile_options(slam_system PRIVATE
        $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall -Wextra -Wpedantic>
        $<$<CXX_COMPILER_ID:MSVC>:/W4>
    )
endif()

# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgproc features2d calib3d highgui imgcodecs)
find_package(Pangolin REQUIRED CONFIG)

target_link_libraries(slam_system PRIVATE
    Eigen3::Eigen
    ${OpenCV_LIBS}
)

target_include_directories(slam_system PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${OpenCV_INCLUDE_DIRS}
)

if(TARGET Pangolin::pangolin)
    target_link_libraries(slam_system PRIVATE Pangolin::pangolin)
elseif(TARGET Pangolin::Pangolin)
    target_link_libraries(slam_system PRIVATE Pangolin::Pangolin)
elseif(TARGET pangolin)
    target_link_libraries(slam_system PRIVATE pangolin)
elseif(DEFINED Pangolin_LIBRARIES)
    target_link_libraries(slam_system PRIVATE ${Pangolin_LIBRARIES})
else()
    message(FATAL_ERROR "Pangolin found but no known exported target / libs")
endif()

if(DEFINED Pangolin_INCLUDE_DIRS)
    target_include_directories(slam_system PRIVATE ${Pangolin_INCLUDE_DIRS})
endif()

message(STATUS "")
message(STATUS "Configuration Summary:")
message(STATUS "  Project:    ${PROJECT_NAME} ${PROJECT_VERSION}")
message(STATUS "  Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  OpenCV:     ${OpenCV_FOUND}")
message(STATUS "  Pangolin:   ${Pangolin_FOUND}")
message(STATUS "  Eigen3:     ${Eigen3_FOUND}")
message(STATUS "")
