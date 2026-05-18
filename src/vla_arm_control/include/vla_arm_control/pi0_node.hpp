#pragma once

#include <string>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"
#include "vla_arm_control/vla_base_node.hpp"

namespace vla_arm_control
{

/**
 * PI-0.5 VLA backend node.
 *
 * Inherits the observation subscriptions, inference thread, and chunk
 * publication from VlaBaseNode. Adds a language instruction subscription
 * and PI-0.5-specific parameters.
 *
 * The runInference() implementation is a STUB. Replace it with the actual
 * PI-0.5 model call once the model integration is ready.
 */
class Pi0Node : public VlaBaseNode
{
public:
  explicit Pi0Node(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());
  // Node name is always "pi0_node" — drives yaml parameter namespace.
  ~Pi0Node() override;

protected:
  trajectory_msgs::msg::JointTrajectory runInference(
    sensor_msgs::msg::Image::ConstSharedPtr rgb_base,
    sensor_msgs::msg::Image::ConstSharedPtr rgb_wrist,
    sensor_msgs::msg::JointState::ConstSharedPtr joint_states) override;

  void adaptOutput(trajectory_msgs::msg::JointTrajectory & traj) override;

private:
  std::string latest_language_instruction_;
  std::mutex  language_mutex_;

  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr language_sub_;

  int    chunk_length_;
  int    action_dim_;
  int    image_width_;
  int    image_height_;
};

}  // namespace vla_arm_control
