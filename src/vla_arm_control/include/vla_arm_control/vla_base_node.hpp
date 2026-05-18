#pragma once

#include <atomic>
#include <mutex>
#include <stdexcept>
#include <thread>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/joint_state.hpp"
#include "trajectory_msgs/msg/joint_trajectory.hpp"

namespace vla_arm_control
{

/**
 * Abstract base class for all VLA (Vision-Language-Action) nodes.
 *
 * Timestamp convention — IMPORTANT:
 *   The action_chunk published on /action_chunk carries header.stamp equal to
 *   the OBSERVATION IMAGE timestamp, NOT the time the chunk was published.
 *
 *   Why: VLA inference takes 100-200ms. By the time a chunk arrives at the
 *   smoother, it is already "old" relative to wall-clock now. If we used
 *   publish time, the smoother would think the chunk was planned for the
 *   current instant and interpolation into it would be off. Using the
 *   observation time lets the smoother correctly map each trajectory point
 *   to an absolute wall-clock moment, even under variable inference latency.
 *
 *   Convention: each trajectory point's time_from_start is relative to
 *   header.stamp (the observation time). Absolute time of point i is:
 *     abs_time_i = chunk.header.stamp + points[i].time_from_start
 *
 * Subclass contract:
 *   1. Declare any additional parameters and subscriptions in your constructor.
 *   2. Call startInferenceThread() at the END of your constructor.
 *   3. Call stopInferenceThread() in your destructor (before any member cleanup).
 *   4. Implement runInference() and adaptOutput() (pure virtual below).
 *   5. Do NOT set header.stamp inside runInference() — the base class sets it.
 */
class VlaBaseNode : public rclcpp::Node
{
public:
  explicit VlaBaseNode(
    const std::string & node_name,
    const rclcpp::NodeOptions & options = rclcpp::NodeOptions())
  : Node(node_name, options)
  {
    declare_parameter("inference_frequency_hz", 5.0);
    declare_parameter("use_wrist_camera", false);
    declare_parameter("joint_names", std::vector<std::string>{});
    declare_parameter("action_chunk_dt", 0.02);

    inference_frequency_hz_ = get_parameter("inference_frequency_hz").as_double();
    use_wrist_camera_ = get_parameter("use_wrist_camera").as_bool();
    joint_names_ = get_parameter("joint_names").as_string_array();
    action_chunk_dt_ = get_parameter("action_chunk_dt").as_double();

    action_chunk_pub_ = create_publisher<trajectory_msgs::msg::JointTrajectory>(
      "/action_chunk", rclcpp::QoS(10));

    rgb_base_sub_ = create_subscription<sensor_msgs::msg::Image>(
      "/observations/rgb_base", rclcpp::QoS(1).best_effort(),
      [this](sensor_msgs::msg::Image::ConstSharedPtr msg) {
        std::lock_guard<std::mutex> lock(obs_mutex_);
        latest_rgb_base_ = msg;
      });

    joint_states_sub_ = create_subscription<sensor_msgs::msg::JointState>(
      "/observations/joint_states", rclcpp::QoS(1).best_effort(),
      [this](sensor_msgs::msg::JointState::ConstSharedPtr msg) {
        std::lock_guard<std::mutex> lock(obs_mutex_);
        latest_joint_states_ = msg;
      });

    if (use_wrist_camera_) {
      rgb_wrist_sub_ = create_subscription<sensor_msgs::msg::Image>(
        "/observations/rgb_wrist", rclcpp::QoS(1).best_effort(),
        [this](sensor_msgs::msg::Image::ConstSharedPtr msg) {
          std::lock_guard<std::mutex> lock(obs_mutex_);
          latest_rgb_wrist_ = msg;
        });
    }
  }

  virtual ~VlaBaseNode()
  {
    stopInferenceThread();
  }

protected:
  // ---- Latest observations (protected: readable by subclass adaptOutput) ----
  sensor_msgs::msg::Image::ConstSharedPtr      latest_rgb_base_;
  sensor_msgs::msg::Image::ConstSharedPtr      latest_rgb_wrist_;   // nullptr if !use_wrist_camera_
  sensor_msgs::msg::JointState::ConstSharedPtr latest_joint_states_;
  std::mutex obs_mutex_;

  rclcpp::Publisher<trajectory_msgs::msg::JointTrajectory>::SharedPtr action_chunk_pub_;

  double inference_frequency_hz_;
  double action_chunk_dt_;
  bool   use_wrist_camera_;
  std::vector<std::string> joint_names_;

  /**
   * Run one inference cycle. Called from the inference thread (not ROS executor).
   *
   * @param rgb_base    Latest base camera image (guaranteed non-null when called).
   * @param rgb_wrist   Latest wrist camera image (nullptr if use_wrist_camera=false).
   * @param joint_states Latest joint state (may be nullptr if no data yet — check before use).
   * @return            Trajectory with points populated. Do NOT set header.stamp.
   *                    Return empty trajectory (no points) to skip publishing this cycle.
   */
  virtual trajectory_msgs::msg::JointTrajectory runInference(
    sensor_msgs::msg::Image::ConstSharedPtr rgb_base,
    sensor_msgs::msg::Image::ConstSharedPtr rgb_wrist,
    sensor_msgs::msg::JointState::ConstSharedPtr joint_states) = 0;

  /**
   * Convert/normalize raw model output into canonical joint positions.
   * Called after runInference(), before publishing. header.stamp is already set.
   *
   * This is where unit conversion, delta accumulation, and joint reordering live.
   * The subclass may read latest_joint_states_ directly (protected member) if needed
   * for delta accumulation — take care that it may have changed since runInference
   * was called; a local copy taken in runInference is safer for determinism.
   */
  virtual void adaptOutput(trajectory_msgs::msg::JointTrajectory & traj) = 0;

  /**
   * Start the inference thread. Call at the END of the derived constructor,
   * after all member variables are initialized.
   */
  void startInferenceThread()
  {
    running_.store(true);
    inference_thread_ = std::thread(&VlaBaseNode::inferenceLoop, this);
  }

  /**
   * Stop the inference thread. Call at the START of the derived destructor,
   * before any member cleanup that the thread might touch.
   */
  void stopInferenceThread()
  {
    running_.store(false);
    if (inference_thread_.joinable()) {
      inference_thread_.join();
    }
  }

private:
  std::atomic<bool> running_{false};
  std::thread inference_thread_;

  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr rgb_base_sub_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr rgb_wrist_sub_;
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_states_sub_;

  void inferenceLoop()
  {
    rclcpp::WallRate rate(inference_frequency_hz_);

    while (running_.load()) {
      // Snapshot the latest observations under lock; do not hold lock during inference.
      sensor_msgs::msg::Image::ConstSharedPtr      rgb_base;
      sensor_msgs::msg::Image::ConstSharedPtr      rgb_wrist;
      sensor_msgs::msg::JointState::ConstSharedPtr joint_states;
      {
        std::lock_guard<std::mutex> lock(obs_mutex_);
        rgb_base     = latest_rgb_base_;
        rgb_wrist    = latest_rgb_wrist_;
        joint_states = latest_joint_states_;
      }

      if (!rgb_base) {
        // No image received yet — wait silently.
        rate.sleep();
        continue;
      }

      trajectory_msgs::msg::JointTrajectory traj;
      try {
        traj = runInference(rgb_base, rgb_wrist, joint_states);
      } catch (const std::exception & e) {
        RCLCPP_WARN(get_logger(), "runInference threw: %s — skipping cycle", e.what());
        rate.sleep();
        continue;
      } catch (...) {
        RCLCPP_WARN(get_logger(), "runInference threw unknown exception — skipping cycle");
        rate.sleep();
        continue;
      }

      if (traj.points.empty()) {
        RCLCPP_WARN(get_logger(), "runInference returned empty trajectory — skipping cycle");
        rate.sleep();
        continue;
      }

      // Set header.stamp to the OBSERVATION timestamp, not now().
      // This is the contract between VLA nodes and the smoother: stamp encodes
      // when the observations were taken, so the smoother can correctly place
      // each trajectory point on the absolute wall-clock timeline.
      traj.header.stamp = rgb_base->header.stamp;

      adaptOutput(traj);

      action_chunk_pub_->publish(traj);

      rate.sleep();
    }
  }
};

}  // namespace vla_arm_control
