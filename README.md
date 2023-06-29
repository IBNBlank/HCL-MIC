# HCL-MIC
This is an implementation of the paper [CL-MIC: A Human-Control-Level Mixed-Initiative Controller in dynamic environment using DRL]()

## Requirement

- python 3.10
- [ROS Melodic](https://wiki.ros.org/melodic)
- [mpi4py](https://mpi4py.readthedocs.io/en/stable/)
- [Stage](http://rtv.github.io/Stage/)
- [PyTorch](http://pytorch.org/)

## How to train

1. Catkin make
```
mkdir -p catkin_ws/src
cd catkin_ws/src
git clone [this repo]
cd ../
catkin_make
source devel/setup.bash
```

2. Pretrain
    1. Collect data
    Set `collect_data` in `python_stage_x.py` (x=0,1,2) to `True` and run the code blow:
    ```
    roslaunch hex_rl_controller stage_x.launch
    rosrun hex_rl_controller bash_stage_x.sh
    ```
    2. Train the model
    ```
    rosrun hex_rl_controller bash_stage_pretrain.sh
    ```

3. Train in Stage0
```
roslaunch hex_rl_controller stage_0.launch
rosrun hex_rl_controller bash_stage_0.sh
```

4. Train in Stage1
```
roslaunch hex_rl_controller stage_1.launch
rosrun hex_rl_controller bash_stage_1.sh
```

5. Train in Stage2
```
roslaunch hex_rl_controller stage_2.launch
rosrun hex_rl_controller bash_stage_2.sh
```

## How to test

* Test in Stage
    ```
    roslaunch hex_rl_controller stage_test.launch
    rosrun hex_rl_controller bash_stage_test.sh
    ```

* Test in Gazebo
    ```
    roslaunch hex_rl_controller stage_gazebo.launch
    rosrun hex_rl_controller bash_gazebo_test.sh
    ```

* Test in real-world
    * Run in the vehicle
        ```
        roslaunch hex_rl_controller bringup_rl_vehicle.launch
        ```
    * Run in the PC
        ```
        roslaunch hex_rl_controller bringup_rl_gui.launch
        ```
