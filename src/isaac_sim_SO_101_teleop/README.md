# isaac_sim_SO_101_teleop

Isaac Sim 5.x bridge for the SO-101 arm with PS4 DualShock joystick control over a Zenoh-bridged ROS2 network.
## Prerequesites
1) Build and source
``` cd isaac_sim_ros2_experimentation ```
```colcon build --packages-select isaac_sim_SO_101_teleop && source install/setup.bash ```
2) Source ros2 envrionment e.g. ```source /opt/ros/jazzy/setup.bash```


### Zenoh Configuration
#### On both
1) Install zenoh router: ```sudo apt install ros-jazzy-rmw-zenoh-cpp```
2) Declare zenoh as the new RMW: ```export RMW_IMPLEMENTATION=rmw_zenoh_cpp``` 


#### On EC2
1) Setup security group: On cloud machine setup a security group rule to restrict inbound traffic on port 7447 to our local IP:

Type: Custom TCP
Port: 7447
Source: <YOUR_HOME_PUBLIC_IP>/32

Note: The /32 means that exact IP only, no range.
Check your current IP with ```curl ifconfig.me``` and update the rule when it changes

2) Launch the Zenoh router: ```ros2 run rmw_zenoh_cpp rmw_zenohd -- --listen tcp/0.0.0.0:7447``` 
- Note: 0.0.0.0 means listen on all interfaces, which ensures it accepts connections coming in via the public IP through AWS NAT.
3) Test connection
- ros2 run demo_nodes_py listener

#### On local machine:
1) Connect to EC2 router explicitly, using 
the [zenoh session config](https://github.com/ros2/rmw_zenoh/blob/rolling/rmw_zenoh_cpp/config/DEFAULT_RMW_ZENOH_SESSION_CONFIG.json5), modify the mode to **client** and the connect:endpoints tp your ec2 IP.
- ```export ZENOH_SESSION_CONFIG_URI=~/Documents/zenoh_session_config.json5``` 
2) Verify connection 
- ```nc -zv <EC2 PUBLIC IP> 7447``, *Connection Refused* output means there is nothing listening on the ec2 instance yet..
- ```ros2 run demo_nodes_py talker```

---

## Notes

- **Wrist roll offset**: `so_101_pub.py` applies a fixed +90° offset to `wrist_roll` so that the leader arm's neutral grip orientation feels natural to the operator. Adjust `wrist_offset_deg` in that file if you need a different offset.

## Procedure

### On Local Machine

export ZENOH_SESSION_CONFIG_URI=~/Documents/zenoh_session_config.json5  # pointing to EC2 IP:7447
export RMW_IMPLEMENTATION=rmw_zenoh_cpp
ros2 run demo_nodes_cpp listener


### Terminal Running Issac Sim

ONLY RUN ONCE PER TERMINAL AS IT IS APPENDED TO THE ENV VAR
```export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/ros/jazzy/lib:/opt/ros/jazzy/opt/zenoh_cpp_vendor/lib```

