#include "vla_arm_control/openvla_oft_node.hpp"

#include "builtin_interfaces/msg/duration.hpp"
#include "trajectory_msgs/msg/joint_trajectory_point.hpp"

namespace vla_arm_control
{

OpenVlaOftNode::OpenVlaOftNode(const rclcpp::NodeOptions & options)
: VlaBaseNode("openvla_oft_node", options)
{
  declare_parameter("language_instruction", std::string(""));
  declare_parameter("image_width", 224);
  declare_parameter("image_height", 224);
  declare_parameter("action_dim", 7);
  declare_parameter("chunk_length", 10);
  declare_parameter("action_scale", 1.0);

  latest_language_instruction_ = get_parameter("language_instruction").as_string();
  image_width_  = get_parameter("image_width").as_int();
  image_height_ = get_parameter("image_height").as_int();
  action_dim_   = get_parameter("action_dim").as_int();
  chunk_length_ = get_parameter("chunk_length").as_int();
  action_scale_ = get_parameter("action_scale").as_double();

  language_sub_ = create_subscription<std_msgs::msg::String>(
    "/observations/language_instruction", rclcpp::QoS(1).transient_local(),
    [this](std_msgs::msg::String::ConstSharedPtr msg) {
      std::lock_guard<std::mutex> lock(language_mutex_);
      latest_language_instruction_ = msg->data;
    });

  RCLCPP_INFO(get_logger(),
    "OpenVlaOftNode ready — chunk_length=%d, action_dim=%d, action_scale=%.3f",
    chunk_length_, action_dim_, action_scale_);

  startInferenceThread();
}

OpenVlaOftNode::~OpenVlaOftNode()
{
  stopInferenceThread();
}

trajectory_msgs::msg::JointTrajectory OpenVlaOftNode::runInference(
  sensor_msgs::msg::Image::ConstSharedPtr rgb_base,
  sensor_msgs::msg::Image::ConstSharedPtr rgb_wrist,
  sensor_msgs::msg::JointState::ConstSharedPtr joint_states)
{
  // =========================================================================
  // STUB — Replace with actual OpenVLA-OFT model call.
  //
  // ASSUMPTION: OpenVLA-OFT is OpenVLA fine-tuned with Orthogonal Fine-Tuning
  //   and parallel action decoding for improved sample efficiency and speed.
  //   Reference: OpenVLA-OFT (2025) — parallel decoding variant.
  //
  // ASSUMPTION: Inputs
  //   - rgb_base    : base camera image, resized to (image_width_, image_height_).
  //                   OpenVLA-OFT does NOT use a wrist camera by default.
  //   - joint_states: current joint positions, used as proprioceptive conditioning.
  //   - language    : task instruction string, processed by the VLM tokenizer.
  //   (rgb_wrist is ignored — use_wrist_camera=false for this node)
  //
  // ASSUMPTION: Output
  //   - Action chunk of shape [chunk_length_, action_dim_] — DELTA actions.
  //   - Each row is a joint-space increment (radians) to apply sequentially.
  //   - action_dim_=7: 6 joint deltas + 1 gripper delta.
  //   - chunk_length_=10 at action_chunk_dt_=0.02s → 0.2-second horizon.
  //   - Raw model output is multiplied by action_scale_ before accumulation.
  //
  // IMPORTANT: adaptOutput() converts these deltas to absolute positions.
  //   The model output stored in trajectory points here must be RAW DELTAS,
  //   not accumulated. adaptOutput() does the accumulation relative to
  //   joint_state_snapshot_ captured below.
  //
  // To implement: load checkpoint, run VLM forward pass with parallel decoding,
  //   populate trajectory points with raw delta values from the model output.
  // =========================================================================
  RCLCPP_INFO_ONCE(get_logger(), "OpenVLA-OFT inference stub — implement model call here");

  (void)rgb_base;
  (void)rgb_wrist;

  // Snapshot current joint positions so adaptOutput() can accumulate deltas.
  // We do this here (in the inference thread, before the model call) so that
  // the base positions correspond to the same observation used for inference.
  if (joint_states && !joint_states->position.empty()) {
    joint_state_snapshot_ = joint_states->position;
  } else {
    joint_state_snapshot_.assign(action_dim_, 0.0);
    RCLCPP_WARN_ONCE(get_logger(),
      "No joint states available — using zero base for delta accumulation");
  }

  std::string instruction;
  {
    std::lock_guard<std::mutex> lock(language_mutex_);
    instruction = latest_language_instruction_;
  }
  (void)instruction;

  // Build a zero-delta trajectory (stub).
  // adaptOutput() will accumulate these zeros relative to the current joint state,
  // resulting in a trajectory that holds the current pose — safe for testing.
  trajectory_msgs::msg::JointTrajectory traj;
  traj.joint_names = joint_names_;

  for (int t = 0; t < chunk_length_; ++t) {
    trajectory_msgs::msg::JointTrajectoryPoint pt;
    // Store raw deltas here — adaptOutput() converts to absolute positions.
    pt.positions.assign(action_dim_, 0.0);

    const double dt_sec = t * action_chunk_dt_;
    pt.time_from_start.sec     = static_cast<int32_t>(dt_sec);
    pt.time_from_start.nanosec = static_cast<uint32_t>(
      (dt_sec - static_cast<int32_t>(dt_sec)) * 1e9);

    traj.points.push_back(std::move(pt));
  }

  return traj;
}

void OpenVlaOftNode::adaptOutput(trajectory_msgs::msg::JointTrajectory & traj)
{
  // Convert delta actions to absolute joint positions by accumulation.
  //
  // The model predicts incremental joint changes (deltas) relative to the
  // current state. To produce a valid trajectory for downstream controllers,
  // we integrate: pos[t] = pos[t-1] + action_scale_ * delta[t]
  //
  // action_scale_ allows tuning the effective magnitude of model outputs
  // without retraining (useful during initial integration).

  if (joint_state_snapshot_.empty()) {
    RCLCPP_WARN(get_logger(), "adaptOutput: no joint state snapshot — skipping delta accumulation");
    return;
  }

  const size_t dof = joint_state_snapshot_.size();
  std::vector<double> current_pos = joint_state_snapshot_;

  for (auto & pt : traj.points) {
    if (pt.positions.size() < dof) {
      RCLCPP_WARN_ONCE(get_logger(),
        "adaptOutput: point has %zu positions but expected %zu — padding with zeros",
        pt.positions.size(), dof);
      pt.positions.resize(dof, 0.0);
    }

    for (size_t j = 0; j < dof; ++j) {
      current_pos[j] += action_scale_ * pt.positions[j];
      pt.positions[j]  = current_pos[j];
    }
  }
}

}  // namespace vla_arm_control

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<vla_arm_control::OpenVlaOftNode>());
  rclcpp::shutdown();
  return 0;
}
