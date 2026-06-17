cmake_minimum_required(VERSION 3.8)
project(pre_trained_vla_test_interfaces)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(sensor_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "srv/GetActionChunk.srv"
  DEPENDENCIES sensor_msgs
)

ament_package()
