#!/usr/bin/env python3

from threading import Lock
from tf2_ros.transform_listener import TransformListener
from tf2_ros.buffer import Buffer
import numpy as np
import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from tf2_geometry_msgs import PoseStamped
from nav_msgs.msg import Odometry
from f110_msgs.msg import WpntArray
from rcl_interfaces.msg import ParameterType, ParameterDescriptor, FloatingPointRange
from rclpy.node import Node
import os
import yaml
from ament_index_python.packages import get_package_share_directory
from parameter_event_handler.parameter_event_handler import ParameterEventHandler

class Controller(Node):
    def __init__(self):
        super().__init__('control_node',
                         allow_undeclared_parameters=True,
                         automatically_declare_parameters_from_overrides=True)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        lab5_path = get_package_share_directory('lab5')
        config_path = os.path.join(lab5_path, 'config', 'solution_params.yaml')
        
        with open(config_path, 'r') as f:
            self.param_dict = yaml.safe_load(f)
            self.param_dict = self.param_dict['controller']['ros__parameters']
            self.m_lookahead = self.param_dict['m_lookahead']
            self.q_lookahead = self.param_dict['q_lookahead']

        # dyn params
        self.param_handler = ParameterEventHandler(self)
        self.callback_handle = self.param_handler.add_parameter_event_callback(
            callback=self.lookahead_param_cb, 
        )

        param_dicts = [{'name' : 'm_lookahead',
                        'default' : self.m_lookahead,
                        'descriptor' : ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE, floating_point_range=[FloatingPointRange(from_value=0.0, to_value=1.0, step=0.001)])},
                       {'name' : 'q_lookahead',
                        'default' : self.q_lookahead,
                        'descriptor' : ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE, floating_point_range=[FloatingPointRange(from_value=-1.0, to_value=1.0, step=0.001)])},
                       ]
        params = self.delcare_dyn_parameters(param_dicts)
        self.set_parameters(params)
        
        self.rate = 40 # rate in hertz 
        self.wheelbase = 0.301 # distance between front and rear axle
        self.lookahead_distance = 1.0 # fixed lookahead distance 
        self.lookahead_point = None # current lookahead point
        self.position_in_map = None # current position in map frame
        self.waypoints_in_map = None # waypoints starting at car's position in map frame
        self.current_latdev = None # lateral deviation from the raceline
        self.current_speed = None # current speed of the car
        
        self.control_msg = AckermannDriveStamped() # empty control message
        self.control_msg.drive.speed = 5.0 # speed is set constant at 5 m/s for LAB 5
        self.control_msg.header.frame_id = 'base_link'

        self.lock = Lock() # lock for the waypoints

        # SUBSCRIBERS
        self.create_subscription(WpntArray,'/global_waypoints',  self.global_waypoints_cb, 10) # waypoints (x, y, v, norm trackbound, s, kappa)
        self.create_subscription(Odometry,'/car_state/odom',  self.speed_cb, 10) # car speed
        self.create_subscription(PoseStamped,'/car_state/pose',  self.car_state_cb, 10) # car position (x, y, theta)
        self.create_subscription(Odometry, '/car_state/frenet/odom',self.frenet_cb, 10) # lat deviation

        # PUBLISHERS
        ctrl_topic_name = '/drive'
        self.drive_pub = self.create_publisher(AckermannDriveStamped, ctrl_topic_name, 10)

        # Block until relevant data is here
        self.wait_for_messages()
        
        self.create_timer(1/self.rate, self.control_loop)
        self.get_logger().info("Controller ready")
        
        
    def wait_for_messages(self):
        self.get_logger().info('Controller Manager waiting for messages...')
        position_in_map_print = False
        waypoints_in_map_print = False
        current_speed_print = False
        current_latdev_print = False

        while self.position_in_map is None or self.waypoints_in_map is None or self.current_speed is None or self.current_latdev is None:
            rclpy.spin_once(self)
            if self.position_in_map is not None and not position_in_map_print:
                self.get_logger().info('Received car position')
                position_in_map_print = True
            if self.waypoints_in_map is not None and not waypoints_in_map_print:
                self.get_logger().info('Received waypoint array')
                waypoints_in_map_print = True
            if self.current_speed is not None and not current_speed_print:
                self.get_logger().info('Received car speed')
                current_speed_print = True
            if self.current_latdev is not None and not current_latdev_print:
                self.get_logger().info('Received lateral deviation')
                current_latdev_print = True

        self.get_logger().info('All required messages received. Continuing...')
        
    def delcare_dyn_parameters(self, param_dicts):
        params = []
        for param_dict in param_dicts:
            param = self.declare_parameter(param_dict['name'], param_dict['default'], param_dict['descriptor'])
            params.append(param)
        return params


    
    #############
    # CALLBACKS #
    #############
    def car_state_cb(self, data: PoseStamped):
        """
        Saves the car position in a numpy array

        Args: 
            data: a PoseStamped message with the car position and much more
        """
        # [task2] save the car position in the map frame of reference
        self.position_in_map = np.empty((1, 2))
        x = data.pose.position.x
        y = data.pose.position.y
        self.position_in_map = np.array([x, y])


    def global_waypoints_cb(self, data: WpntArray):
        """
        Saves the global waypoints in a numpy array

        Args:
            data: A WpntArray message with the global trajectory waypoints in map frame
        """
        with self.lock:
            # [task2] save the global waypoints in the map frame of reference
            self.waypoints_in_map = np.empty((0,2))
            self.speed_waypoints = np.empty((0,1))
            self.waypoints_in_map = np.array([
                    [waypoint.x_m, waypoint.y_m] for waypoint in data.wpnts
                ])
            self.speed_waypoints = np.array([
                    waypoint.vx_mps for waypoint in data.wpnts
                ])

    def speed_cb(self, data: Odometry):
        # [task2] save the car's longitudinal speed 
        self.current_speed = 0
        self.current_speed = data.twist.twist.linear.x

    def frenet_cb(self, data: Odometry):
        """
        Save the lateral deviation from the /car_state/frenet topic. 
        Messages from the topic store (current advancement, lateral deviation) on (x, y) of an 
        Odometry message. They are calculated from the global trajectory.
        """
        self.current_latdev = data.pose.pose.position.y

    def lookahead_param_cb(self, parameter_event):
        if parameter_event.node != "/controller":
            return
        
        self.m_lookahead = self.get_parameter('m_lookahead').value
        self.q_lookahead = self.get_parameter('q_lookahead').value

    #############
    # MAIN LOOP #
    #############
    def control_loop(self):
        if self.waypoints_in_map.shape[0] > 2:
            # 1. Set the car's velocity
            with self.lock:
                self.control_msg.drive.speed = self.calc_speed()

            # 2. FIND THE POINT TO PURSUIT
            with self.lock:
                self.lookahead_point = self.find_pursuit_wpnt()
            
            # 3. OBTAIN STEERING WITH PURE PURSUIT LOGIC
            lookahead_point_in_baselink = self.map_to_baselink(self.lookahead_point)
            steering_angle = self.get_actuation(lookahead_point_in_baselink)

            # 3. send steering command to mux
            self.control_msg.header.stamp = self.get_clock().now().to_msg()
            self.control_msg.drive.steering_angle = steering_angle
            self.drive_pub.publish(self.control_msg)
            
            # leave this here, will be needed in lab 6
            self.on_end()
        

    ####################
    # HELPER FUNCTIONS #
    ####################
    def map_to_baselink(self, point: np.array):
        """
        Converts a point from the map frame of reference to the base_link frame of reference

        Args:
            point: a np.array in map frame with shape (2,)

        Returns:
            point_bl: a np.array in base_link frame with shape (2,)
        """
        # [task5] transform the coordinate
        t_map = self.tf_buffer.lookup_transform('map', 'car_state/base_link', rclpy.time.Time())
        
        posestamped_in_map = PoseStamped()
        posestamped_in_map.header.stamp = t_map.header.stamp
        posestamped_in_map.header.frame_id = 'map'
        posestamped_in_map.pose.position.x = point[0]
        posestamped_in_map.pose.position.y = point[1]
        posestamped_in_map.pose.position.z = 0.0
        
        pose_in_baselink = self.tf_buffer.transform(posestamped_in_map, "car_state/base_link")
        point_bl = np.array([
            pose_in_baselink.pose.position.x,
            pose_in_baselink.pose.position.y
        ])
        
        return point_bl

    def find_pursuit_wpnt(self):
        """
        Calculates the waypoint at a certain distance in front of the car

        Returns:
            waypoint as numpy array at a certain distance in front of the car
        """
        self.lookahead_distance = self.calc_lookahead_distance()

        # [task4] find the lookahead point
        # 1. find nearest waypoint idx to current position
        nearest_wpt_idx = np.argmin(np.linalg.norm(self.waypoints_in_map - self.position_in_map, axis=1))
        
        # 2. find lookahead point, only looking in points forward
        looking_zone_len = 20
        if nearest_wpt_idx < len(self.waypoints_in_map)-looking_zone_len:
            idx_waypoint_at_distance = np.argmin(
                    np.abs(
                            np.linalg.norm(self.waypoints_in_map[nearest_wpt_idx:(nearest_wpt_idx+looking_zone_len)] - self.position_in_map, axis = 1)-self.lookahead_distance
                        )
                )
            global_idx = nearest_wpt_idx+idx_waypoint_at_distance
        else:
            # consider wrapping!
            waypoints_search = np.concatenate([self.waypoints_in_map[nearest_wpt_idx:nearest_wpt_idx+looking_zone_len, :2], self.waypoints_in_map[:looking_zone_len, :2]])
            idx_waypoint_at_distance = np.argmin(
                    np.abs(
                            np.linalg.norm(waypoints_search - self.position_in_map, axis = 1)-self.lookahead_distance
                        )
                )
            global_idx = (nearest_wpt_idx+idx_waypoint_at_distance)%len(self.waypoints_in_map)
        return np.array(self.waypoints_in_map[global_idx])

    def get_actuation(self, lookahead_point):
        """
        Calculates steering angle

        Args: 
            lookahead_point: np.array with the lookahead point in base_link coords

        Returns:
            steering angle: steering angle such that car follows arc to lookahead point
        """
        # [task5] find the steering angle according to the relationships described in the pdf
        
        waypoint_y = lookahead_point[1]
        if np.abs(waypoint_y) < 1e-6:
            return 0
        radius = np.linalg.norm(lookahead_point)**2 / (2.0 * waypoint_y)
        steering_angle = np.arctan(self.wheelbase / radius)

        return steering_angle
    
    def calc_speed(self):
        """
        Calculates the speed to command

        Returns: 
            speed: the float commanded speed
        """
        speed = 0

        # [task3] use the speed of the nearest waypoint
        nearest_wpt_idx = np.argmin(np.linalg.norm(self.waypoints_in_map - self.position_in_map, axis = 1))
        speed = 0.5*self.speed_waypoints[nearest_wpt_idx]

        # [task8] consider a further waypoint to account for the delays, increase speed
        final_idx = int(nearest_wpt_idx+self.current_speed*0.25)%self.waypoints_in_map.shape[0]
        speed = self.speed_waypoints[final_idx]
        speed *= 0.75

        return speed
    
    def calc_lookahead_distance(self):
        """
        Computes the lookahead distance

        Returns:
            lookahead_distance: the float for the lookahead distance in meters
        """
        lookahead_distance = 1

        # [task7] adapt the lookahead distance based on the speed of the car
        lookahead_distance = self.q_lookahead + self.m_lookahead*self.current_speed # works with q=0.4 m=0.2

        # [task6] adapt the lookahead distance based on the lateral deviation 
        lookahead_distance = np.clip(lookahead_distance, np.sqrt(3)*abs(self.current_latdev), 5) 

        return lookahead_distance
    
    def on_end(self):
        pass

def main():
    rclpy.init()
    node = Controller()
    rclpy.spin(node)
    rclpy.shutdown()

