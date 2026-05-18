#include "vla_arm_control/pi0_node.hpp"

#include "builtin_interfaces/msg/duration.hpp"
#include "trajectory_msgs/msg/joint_trajectory_point.hpp"

namespace vla_arm_control
{

Pi0Node::Pi0Node(const rclcpp::NodeOptions & options)
: VlaBaseNode("pi0_node", options)
{
  declare_parameter("language_instruction", std::string(""));
  declare_parameter("image_width", 224);
  declare_parameter("image_height", 224);
  declare_parameter("action_dim", 7);
  declare_parameter("chunk_length", 50);

  latest_language_instruction_ = get_parameter("language_instruction").as_string();
  image_width_  = get_parameter("image_width").as_int();
  image_height_ = get_parameter("image_height").as_int();
  action_dim_   = get_parameter("action_dim").as_int();
  chunk_length_ = get_parameter("chunk_length").as_int();

  language_sub_ = create_subscription<std_msgs::msg::String>(
    "/observations/language_instruction", rclcpp::QoS(1).transient_local(),
    [this](std_msgs::msg::String::ConstSharedPtr msg) {
      std::lock_guard<std::mutex> lock(language_mutex_);
      latest_language_instruction_ = msg->data;
    });

  RCLCPP_INFO(get_logger(),
    "Pi0Node ready — chunk_length=%d, action_dim=%d, use_wrist=%s",
    chunk_length_, action_dim_, use_wrist_camera_ ? "true" : "false");

  // Must be last: starts the inference thread, which may call runInference()
  // immediately, so all member variables must be initialized first.
  startInferenceThread();
}

Pi0Node::~Pi0Node()
{
  // Stop the thread before any member variables are destroyed.
  stopInferenceThread();
}

trajectory_msgs::msg::JointTrajectory Pi0Node::runInference(
  sensor_msgs::msg::Image::ConstSharedPtr rgb_base,
  sensor_msgs::msg::Image::ConstSharedPtr rgb_wrist,
  sensor_msgs::msg::JointState::ConstSharedPtr joint_states)
{
  // =========================================================================
  // STUB — Replace with actual PI-0.5 model call.
  //
  // ASSUMPTION: PI-0.5 uses conditional flow matching to generate action chunks.
  //   Reference: Physical Intelligence π0 (2024) — flow matching over action space.
  //
  // ASSUMPTION: Inputs
  //   - rgb_base    : base camera image, resized to (image_width_, image_height_)
  //                   before passing to the model. Currently raw sensor_msgs::Image.
  //   - rgb_wrist   : wrist camera image (use_wrist_camera=true by default for PI-0.5).
  //                   Same preprocessing as rgb_base.
  //   - joint_states: current joint positions/velocities as conditioning signal.
  //   - language    : string task description, tokenized and embedded by the model.
  //
  // ASSUMPTION: Output
  //   - Action chunk of shape [chunk_length_, action_dim_] floats.
  //   - action_dim_=7: 6 DoF joint angles (radians, absolute) + 1 gripper (0-1 normalized).
  //   - chunk_length_=50 at action_chunk_dt_=0.02s → 1-second horizon.
  //
  // ASSUMPTION: Model expects images in RGB uint8 format at 224x224.
  //   Convert from sensor_msgs::Image (encoding in msg->encoding) before calling.
  //
  // To implement: load the model checkpoint, run tokenization + forward pass,
  //   populate trajectory points with the resulting joint angle sequence.
  // =========================================================================
  RCLCPP_INFO_ONCE(get_logger(), "PI-0.5 inference stub — implement model call here");

  (void)rgb_base;
  (void)rgb_wrist;
  (void)joint_states;

  std::string instruction;
  {
    std::lock_guard<std::mutex> lock(language_mutex_);
    instruction = latest_language_instruction_;
  }
  (void)instruction;

  // Build a zero-filled trajectory of the correct shape so downstream pipeline
  // (SmootherNode, controllers) can be tested end-to-end without a real model.
  trajectory_msgs::msg::JointTrajectory traj;
  traj.joint_names = joint_names_;

  for (int t = 0; t < chunk_length_; ++t) {
    trajectory_msgs::msg::JointTrajectoryPoint pt;
    pt.positions.assign(action_dim_, 0.0);

    // time_from_start is relative to header.stamp (observation time).
    // The base class will set header.stamp after this function returns.
    const double dt_sec = t * action_chunk_dt_;
    pt.time_from_start.sec     = static_cast<int32_t>(dt_sec);
    pt.time_from_start.nanosec = static_cast<uint32_t>(
      (dt_sec - static_cast<int32_t>(dt_sec)) * 1e9);

    traj.points.push_back(std::move(pt));
  }

  return traj;
}

void Pi0Node::adaptOutput(trajectory_msgs::msg::JointTrajectory & traj)
{
  // PI-0.5 outputs absolute joint angles — no accumulation or unit conversion needed.
  // Just validate that the joint count matches the configured joint_names.
  if (joint_names_.empty()) {
    return;  // No validation possible without joint names.
  }

  for (auto & pt : traj.points) {
    if (pt.positions.size() != joint_names_.size()) {
      RCLCPP_WARN_ONCE(get_logger(),
        "PI-0.5 output has %zu DoF but joint_names has %zu — trajectory may be malformed",
        pt.positions.size(), joint_names_.size());
      break;
    }
  }
}

}  // namespace vla_arm_control

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<vla_arm_control::Pi0Node>());
  rclcpp::shutdown();
  return 0;
}
