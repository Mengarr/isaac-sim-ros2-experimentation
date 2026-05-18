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
1) Setup security group: On cloud machine setup a secuirty group rule to restrict inbound traffic on port 7447 to our local IP:

Type: Custom TCP
Port: 7447
Source: <YOUR_HOME_PUBLIC_IP>/32

Note: The /32 means that exact IP only, no range.
Check your current IP with ```curl ifconfig.me``` and update the rule when it changes

2) Launch the Zenoh router: ```ros2 run rmw_zenoh_cpp rmw_zenohd```
3) Run nodes

#### On local machine:
1) Connect to EC2 router explicitly: ```export ZENOH_CONFIG_OVERRIDE='{"mode":"client","connect":{"endpoints":["tcp/<EC2_PUBLIC_IP>:7447"]}}'```
2) Run nodes

---

## Proceedure

### On Local Machine
```ros2 launch isaac_sim_SO_101_teleop joy_pub.launch.py```

### On Sim Machine (EC2 instance)
Run script via Isaac Sim's bundled python interpreter: 
```./python.sh ~/.local/share/ov/pkg/isaac-sim-5.x.x/python.sh sim_launcher.py```
