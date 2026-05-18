#pragma once

#include <string>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"
#include "vla_arm_control/vla_base_node.hpp"

namespace vla_arm_control
{

/**
 * OpenVLA-OFT VLA backend node.
 *
 * OpenVLA-OFT produces DELTA actions (joint-space increments), not absolute
 * positions. adaptOutput() accumulates these deltas relative to the current
 * joint state to produce an absolute trajectory before publishing.
 *
 * The runInference() implementation is a STUB. Replace it with the actual
 * OpenVLA-OFT model call once the model integration is ready.
 */
class OpenVlaOftNode : public VlaBaseNode
{
public:
  explicit OpenVlaOftNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());
  ~OpenVlaOftNode() override;

protected:
  trajectory_msgs::msg::JointTrajectory runInference(
    sensor_msgs::msg::Image::ConstSharedPtr rgb_base,
    sensor_msgs::msg::Image::ConstSharedPtr rgb_wrist,
    sensor_msgs::msg::JointState::ConstSharedPtr joint_states) override;

  /**
   * Accumulate delta actions relative to current joint positions.
   * On entry, each trajectory point holds raw delta values from the model.
   * On exit, each trajectory point holds absolute joint positions.
   */
  void adaptOutput(trajectory_msgs::msg::JointTrajectory & traj) override;

private:
  std::string latest_language_instruction_;
  std::mutex  language_mutex_;

  // Snapshot of joint states taken at the start of runInference, used in adaptOutput.
  // Protected by the fact that adaptOutput is called synchronously after runInference
  // in the same inference thread — no additional lock needed.
  std::vector<double> joint_state_snapshot_;

  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr language_sub_;

  int    chunk_length_;
  int    action_dim_;
  int    image_width_;
  int    image_height_;
  double action_scale_;
};

}  // namespace vla_arm_control
