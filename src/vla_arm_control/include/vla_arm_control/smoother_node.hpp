#pragma once

#include <deque>
#include <mutex>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "trajectory_msgs/msg/joint_trajectory.hpp"

namespace vla_arm_control
{

class SmootherNode : public rclcpp::Node
{
public:
  explicit SmootherNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());

private:
  struct ChunkEntry
  {
    trajectory_msgs::msg::JointTrajectory chunk;
    rclcpp::Time receive_time;
  };

  void onActionChunk(trajectory_msgs::msg::JointTrajectory::ConstSharedPtr msg);
  void onTimer();

  rclcpp::Subscription<trajectory_msgs::msg::JointTrajectory>::SharedPtr action_chunk_sub_;
  rclcpp::Publisher<trajectory_msgs::msg::JointTrajectory>::SharedPtr    joint_cmd_pub_;
  rclcpp::TimerBase::SharedPtr timer_;

  std::deque<ChunkEntry> chunk_buffer_;
  std::mutex buffer_mutex_;

  double ensemble_lambda_;
  int    max_chunk_buffer_size_;
  std::vector<std::string> joint_names_;
};

}  // namespace vla_arm_control
