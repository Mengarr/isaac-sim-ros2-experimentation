#include "vla_arm_control/smoother_node.hpp"

#include <algorithm>
#include <cmath>
#include <chrono>

#include "builtin_interfaces/msg/duration.hpp"
#include "trajectory_msgs/msg/joint_trajectory_point.hpp"

namespace vla_arm_control
{

SmootherNode::SmootherNode(const rclcpp::NodeOptions & options)
: Node("smoother_node", options)
{
  declare_parameter("execution_frequency_hz", 50.0);
  declare_parameter("max_chunk_buffer_size", 3);
  declare_parameter("ensemble_lambda", 2.0);
  declare_parameter("joint_names", std::vector<std::string>{});

  const double freq      = get_parameter("execution_frequency_hz").as_double();
  max_chunk_buffer_size_ = get_parameter("max_chunk_buffer_size").as_int();
  ensemble_lambda_       = get_parameter("ensemble_lambda").as_double();
  joint_names_           = get_parameter("joint_names").as_string_array();

  action_chunk_sub_ = create_subscription<trajectory_msgs::msg::JointTrajectory>(
    "/action_chunk", rclcpp::QoS(10),
    std::bind(&SmootherNode::onActionChunk, this, std::placeholders::_1));

  joint_cmd_pub_ = create_publisher<trajectory_msgs::msg::JointTrajectory>(
    "/joint_commands", rclcpp::QoS(10));

  const auto period = std::chrono::duration<double>(1.0 / freq);
  timer_ = create_wall_timer(period, std::bind(&SmootherNode::onTimer, this));

  RCLCPP_INFO(get_logger(), "SmootherNode ready — %.0f Hz, lambda=%.2f, buffer=%d",
    freq, ensemble_lambda_, max_chunk_buffer_size_);
}

void SmootherNode::onActionChunk(
  trajectory_msgs::msg::JointTrajectory::ConstSharedPtr msg)
{
  if (msg->points.empty()) {
    RCLCPP_WARN(get_logger(), "Received empty action chunk — ignoring");
    return;
  }

  std::lock_guard<std::mutex> lock(buffer_mutex_);

  ChunkEntry entry;
  entry.chunk        = *msg;
  entry.receive_time = this->now();

  chunk_buffer_.push_back(std::move(entry));

  // Drop oldest chunk when buffer exceeds capacity.
  while (static_cast<int>(chunk_buffer_.size()) > max_chunk_buffer_size_) {
    chunk_buffer_.pop_front();
  }
}

void SmootherNode::onTimer()
{
  const rclcpp::Time now = this->now();

  std::lock_guard<std::mutex> lock(buffer_mutex_);

  // ---- Evict stale chunks ------------------------------------------------
  // A chunk is stale when its last planned action moment has already passed.
  // Last action moment = observation_stamp + time_from_start of final point.
  chunk_buffer_.erase(
    std::remove_if(
      chunk_buffer_.begin(), chunk_buffer_.end(),
      [&now](const ChunkEntry & e) {
        const auto & last_pt = e.chunk.points.back();
        // Reconstruct the absolute time of the last trajectory point.
        // header.stamp is the observation time (see VlaBaseNode timestamp convention).
        const rclcpp::Time last_abs =
          rclcpp::Time(e.chunk.header.stamp) +
          rclcpp::Duration(last_pt.time_from_start);
        return last_abs < now;
      }),
    chunk_buffer_.end());

  if (chunk_buffer_.empty()) {
    return;  // No valid chunks — withhold command rather than repeat stale one.
  }

  // ---- Interpolate each chunk at `now` and compute ensemble weights -------
  //
  // Temporal ensemble: weight recent chunks (by receive_time) more heavily
  // using an exponential decay. This smooths transitions when a new chunk
  // arrives mid-execution by blending it gradually with the prior chunk.
  //
  //   weight_i = exp(-lambda * (now - receive_time_i))
  //
  // where receive_time is wall-clock arrival at this node, NOT observation time.

  const size_t dof = joint_names_.empty()
    ? chunk_buffer_.front().chunk.points.front().positions.size()
    : joint_names_.size();

  std::vector<double> weighted_sum(dof, 0.0);
  double weight_total = 0.0;

  for (const auto & entry : chunk_buffer_) {
    const auto & pts = entry.chunk.points;
    const rclcpp::Time obs_stamp(entry.chunk.header.stamp);

    // Find the two trajectory points that straddle `now`.
    // Absolute time of point i = obs_stamp + pts[i].time_from_start
    int lo = -1;
    for (size_t i = 0; i + 1 < pts.size(); ++i) {
      const rclcpp::Time t_i   = obs_stamp + rclcpp::Duration(pts[i].time_from_start);
      const rclcpp::Time t_ip1 = obs_stamp + rclcpp::Duration(pts[i + 1].time_from_start);
      if (t_i <= now && now <= t_ip1) {
        lo = static_cast<int>(i);
        break;
      }
    }

    if (lo < 0) {
      // `now` is outside the valid range of this chunk — skip it.
      // This can happen transiently while a new chunk is arriving.
      continue;
    }

    const rclcpp::Time t_lo = obs_stamp + rclcpp::Duration(pts[lo].time_from_start);
    const rclcpp::Time t_hi = obs_stamp + rclcpp::Duration(pts[lo + 1].time_from_start);
    const double span = (t_hi - t_lo).seconds();

    // Guard against zero-length segments.
    const double alpha = (span > 1e-9) ? ((now - t_lo).seconds() / span) : 0.0;

    // Linearly interpolate positions between the two surrounding points.
    const auto & pos_lo = pts[lo].positions;
    const auto & pos_hi = pts[lo + 1].positions;

    if (pos_lo.size() < dof || pos_hi.size() < dof) {
      RCLCPP_WARN_ONCE(get_logger(),
        "Chunk has fewer positions (%zu) than expected DoF (%zu) — skipping chunk",
        std::min(pos_lo.size(), pos_hi.size()), dof);
      continue;
    }

    // Ensemble weight: more recently received chunks get higher weight.
    const double dt     = (now - entry.receive_time).seconds();
    const double weight = std::exp(-ensemble_lambda_ * dt);

    for (size_t j = 0; j < dof; ++j) {
      weighted_sum[j] += weight * (pos_lo[j] + alpha * (pos_hi[j] - pos_lo[j]));
    }
    weight_total += weight;
  }

  if (weight_total < 1e-12) {
    return;  // All chunks were out of range this tick.
  }

  // ---- Publish the blended single-point command --------------------------
  trajectory_msgs::msg::JointTrajectory cmd;
  cmd.header.stamp  = now;
  cmd.joint_names   = joint_names_.empty()
    ? chunk_buffer_.front().chunk.joint_names
    : joint_names_;

  trajectory_msgs::msg::JointTrajectoryPoint pt;
  pt.positions.resize(dof);
  for (size_t j = 0; j < dof; ++j) {
    pt.positions[j] = weighted_sum[j] / weight_total;
  }
  // time_from_start = 0 signals "execute immediately"
  pt.time_from_start.sec     = 0;
  pt.time_from_start.nanosec = 0;

  cmd.points.push_back(std::move(pt));
  joint_cmd_pub_->publish(cmd);
}

}  // namespace vla_arm_control

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<vla_arm_control::SmootherNode>());
  rclcpp::shutdown();
  return 0;
}
