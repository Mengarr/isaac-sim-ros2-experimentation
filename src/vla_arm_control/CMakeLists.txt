cmake_minimum_required(VERSION 3.8)
project(vla_arm_control)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(trajectory_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)

set(DEPS rclcpp trajectory_msgs sensor_msgs std_msgs)

# smoother_node — temporal ensemble smoother, runs at 50Hz
add_executable(smoother_node src/smoother_node.cpp)
target_include_directories(smoother_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
ament_target_dependencies(smoother_node ${DEPS})

# pi0_node — PI-0.5 VLA backend (stub)
add_executable(pi0_node src/pi0_node.cpp)
target_include_directories(pi0_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
ament_target_dependencies(pi0_node ${DEPS})

# openvla_oft_node — OpenVLA-OFT VLA backend (stub)
add_executable(openvla_oft_node src/openvla_oft_node.cpp)
target_include_directories(openvla_oft_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
ament_target_dependencies(openvla_oft_node ${DEPS})

install(TARGETS smoother_node pi0_node openvla_oft_node
  DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY include/
  DESTINATION include)

install(DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME})

ament_package()
