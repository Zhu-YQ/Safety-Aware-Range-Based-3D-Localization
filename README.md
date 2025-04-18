# Safety-Aware Range-Based 3D Localization
Submitted to IROS2025. The full version of code will be released if the paper is accepted.

## 1. Prerequisites
### 1.1 Ubuntu and ROS
- Ubuntu 20.04
- ROS Noetic

### 1.2 Dependencies

- Eigen3

- OpenCV4
- G2O（[compatible version](https://github.com/RainerKuemmerle/g2o/tree/9b41a4ea5ade8e1250b9c1b279f3a9c098811b5a)）



## 2. Build

```
cd catkin_workspace/src
git clone https://github.com/Zhu-YQ/Safety-Aware-Range-Based-3D-Localization.git 
cd ../
catkin_make
```



## 3. Run

A rosbag for simulation is provided. You can play it directly to check the effect of this system.

```
source devel/setup.bash
roslaunch uwb_loc run_sim.launch
```

