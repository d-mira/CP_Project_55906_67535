cmake_minimum_required(VERSION 3.16)

project(project)

set(CMAKE_CXX_STANDARD 17)

###############
# libWB
###############

set(BUILD_LIBWB_LIBRARY ON CACHE BOOL "Turn on build of libwb library")
add_subdirectory(libwb)
include_directories(libwb)

###############
# OpenMp
###############

find_package(OpenMP REQUIRED)

if(OPENMP_FOUND)
    find_package(OpenMP)
    add_executable(project_par
            src/main.cpp
            src/histogram_eq_par.cpp
    )
    target_compile_definitions(project_par PRIVATE PROJECT_NAME="project_par")
    target_link_libraries(project_par PUBLIC OpenMP::OpenMP_C stdc++ m)
    target_link_libraries(project_par PUBLIC wb)
else()
    message(FATAL_ERROR "OpenMP not found")
endif()

###############
# CUDA
###############

find_package(CUDA REQUIRED)

if(CUDA_FOUND)
    message("Cuda found")
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -O3 --std=c++17")
    include_directories(${CUDA_INCLUDE_DIRS})


    add_executable(project_cuda
            src/main.cpp
            src/histogram_eq_cud.cu
    )
    target_compile_definitions(project_cuda PRIVATE PROJECT_NAME="project_cuda")
    target_link_libraries(project_cuda PUBLIC ${CUDA_LIBRARIES} wb)
else()
    message(FATAL_ERROR "CUDA not found")
endif()

###############
# project
###############

include_directories(include)

add_executable(project
        src/main.cpp
        src/histogram_eq.cpp
)
target_compile_definitions(project PRIVATE PROJECT_NAME="project")
target_link_libraries(project wb)

##############################################################
# Tests
##############################################################

enable_testing()

#############
# Download and unpack googletest at configure time
#############

set (GTEST_HOME ${CMAKE_CURRENT_BINARY_DIR}/googletest)

message ("-- Loading Google Test from ${GTEST_HOME}")

configure_file(${PROJECT_SOURCE_DIR}/cmake/CMakeLists.googletest.in ${GTEST_HOME}/download/CMakeLists.txt)

execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
        RESULT_VARIABLE result
        WORKING_DIRECTORY ${GTEST_HOME}/download )
if(result)
    message(FATAL_ERROR "CMake step for googletest failed: ${result}")
endif()

execute_process(COMMAND ${CMAKE_COMMAND} --build .
        RESULT_VARIABLE result
        WORKING_DIRECTORY ${GTEST_HOME}/download )
if(result)
    message(FATAL_ERROR "Build step for googletest failed: ${result}")
endif()

# Prevent overriding the parent project's compiler/linker settings on Windows
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

# Add googletest directly to our build. This defines the gtest and gtest_main targets.
add_subdirectory(${GTEST_HOME}/src
        ${GTEST_HOME}/build
        EXCLUDE_FROM_ALL)

set (GTEST_BOTH_LIBRARIES gtest_main gtest)

add_subdirectory (test)
