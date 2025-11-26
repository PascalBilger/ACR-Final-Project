#!/usr/bin/env python3
from typing import List

import numpy as np
import rclpy
from tf2_geometry_msgs import PointStamped
from std_msgs.msg import Bool
from f110_msgs.msg import Obstacle, ObstacleArray, Wpnt, WpntArray
from nav_msgs.msg import Odometry
from scipy.interpolate import InterpolatedUnivariateSpline as Spline
from visualization_msgs.msg import Marker, MarkerArray
from frenet_conversion.frenet_converter import FrenetConverter
from rclpy.node import Node

class ObstacleSpliner(Node):
    """
    This class implements a ROS node that performs splining around obstacles.

    It subscribes to the following topics:
        - `/clicked_point`: Publishes information about the obstacles in rviz.
        - `/erase_btn_click`: Publishes when the erase button is clicked (simply to clear the obstacle array).
        - `/global_waypoints`: Publishes the global waypoints.
    
    The node publishes the following topics:
        - `/planner/avoidance/markers`: Publishes spline markers.
        - `/planner/avoidance/wpnts`: Publishes splined waypoints.
    """

    def __init__(self):
        """
        Initialize the node, subscribe to topics, and create publishers and service proxies.
        """
        # Initialize the node
        super().__init__('spliner',
                         allow_undeclared_parameters=True,
                         automatically_declare_parameters_from_overrides=True)
        
        # initialize the instance variable
        self.obstacles = ObstacleArray()
        self.gb_vmax = None
        self.gb_max_idx = None
        self.gb_max_s = None
        self.waypoints = None
        self.frenet_state = Odometry()
        self.gb_wpnts = WpntArray()
        self.lookahead = 10  # in meters [m]
        self.evasion_dist = 0.65 # in meters [m]
        self.obs_traj_tresh = 0.8 # in meters [m]
        self.spline_bound_mindist = 0.2 # in meters [m]

        # Create subscriptions
        self.create_subscription(PointStamped, '/clicked_point', self.obs_cb, 10)
        self.create_subscription(Bool, '/erase_btn_click', self.erase_btn_cb, 10)
        self.create_subscription(Odometry, '/car_state/frenet/odom', self.state_cb, 10)
        self.create_subscription(WpntArray, '/global_waypoints', self.gb_cb, 10)
        
        # Create publishers
        self.mrks_pub = self.create_publisher(MarkerArray, "/planner/avoidance/markers", 10)
        self.evasion_pub = self.create_publisher(WpntArray, "/planner/avoidance/otwpnts", 10)
        self.closest_obs_pub = self.create_publisher(Marker, "/planner/avoidance/considered_OBS", 10)
        
        #Frenet Converter
        self.converter = self.initialize_converter()
        
        # Wait for critical Messages and services
        self._logger.info("[Spliner] Waiting for messages...")
        # Already waited for /global_waypoints in initialize_converter
        # Wait for /car_state/odom
        while rclpy.ok() and self.frenet_state is None:
            rclpy.spin_once(self)
            
        self._logger.info("[Spliner] Ready!")
        
        self.rate = 20 # Hz
        self.create_timer(1/self.rate, self.loop)
        
    # Callback for obstacle topic
    def obs_cb(self, data: PointStamped):
        if self.waypoints is not None:
            x_cart = data.point.x
            y_cart = data.point.y

            # Convert to frenet
            s, d = self.converter.get_frenet([x_cart], [y_cart])

            # Create obstacle msg object
            car_width = 0.3  # in meters [m]
            car_length = 0.5  # in meters [m]
            obs = Obstacle()
            obs.s_center = s[0]
            obs.s_start = s[0] - car_length/2
            obs.s_end = s[0] + car_length/2
            obs.d_center = d[0]
            obs.d_left = d[0] - car_width/2
            obs.d_right = d[0] + car_width/2
            self.obstacles.obstacles.append(obs)

            self.get_logger().info('Adding obs to the obstacle list!')        
        else:
            self.get_logger().info('[Spliner] No waypoints yet, waiting for /global_waypoints [WpntArray]')
            pass
        
    def state_cb(self, data: Odometry):
        self.frenet_state = data

    def erase_btn_cb(self, data: Bool):
        if self.obstacles is not None:
            self.obstacles.obstacles = []
            print('[Spliner] Erasing obstacles!')
        else:
            pass
        
    # Callback for global waypoint topic
    def gb_cb(self, data: WpntArray):
        self.waypoints = np.array([[wpnt.x_m, wpnt.y_m, wpnt.psi_rad] for wpnt in data.wpnts])
        self.gb_wpnts = data
        if self.gb_vmax is None:
            self.gb_vmax = np.max(np.array([wpnt.vx_mps for wpnt in data.wpnts]))
            self.gb_max_idx = data.wpnts[-1].id
            self.gb_max_s = data.wpnts[-1].s_m
            
    def initialize_converter(self) -> FrenetConverter:
        """
        Initialize the FrenetConverter object
        """
        # wait for the global waypoints
        while rclpy.ok() and self.waypoints is None:
            rclpy.spin_once(self)

        # Initialize the FrenetConverter object
        converter = FrenetConverter(self.waypoints[:, 0], self.waypoints[:, 1], self.waypoints[:, 2])
        self._logger.info("[Spliner] initialized FrenetConverter object")
        return converter

    # Main loop that runs until ros shutdown
    def loop(self):
        # Sample data
        obstacles = self.obstacles
        gb_scaled_wpnts = self.gb_wpnts.wpnts
        wpnts = WpntArray()
        mrks = MarkerArray()
        frenet_state = self.frenet_state

        # If obs then do splining around it
        if len(obstacles.obstacles) > 0:
            wpnts, mrks = self._do_spline(obstacles=obstacles, gb_wpnts=gb_scaled_wpnts, frenet_state=frenet_state, wpnts=wpnts)
        # Else delete spline markers
        else:
            del_mrk = Marker()
            del_mrk.header.stamp = self.get_clock().now().to_msg()
            del_mrk.action = Marker.DELETEALL
            mrks.markers.append(del_mrk)

        # Publish wpnts and markers
        self.mrks_pub.publish(mrks)
        self.evasion_pub.publish(wpnts)
        
    #########################################Helpers#############################################

    def _obs_filtering(self, frenet_state: Odometry, obstacles: ObstacleArray) -> List[Obstacle]:
        """
        Filters obstacles based on their relevance to the current raceline and the lookahead distance.
        
        Parameters:
        - frenet_state (Odometry): The current state of the vehicle in Frenet coordinates.
        - obstacles (ObstacleArray): Array of obstacles detected in the environment.
        
        Returns:
        - List[Obstacle]: List of obstacles that are both within a specific threshold of the raceline and 
                          within the lookahead distance in front of the car.
        """
        # 1:
        # 1. Initialize the close_obs list.
        # 2. Loop through the obstacles in the obstacles.obstacles list.
        # 3. For each obstacle, determine if its distance from the raceline (`obs.d_center`) is within `self.obs_traj_tresh`.
        # 4. For each obstacle within the raceline threshold, check if its s-coordinate (`obs.s_center`) is within `self.lookahead` from the vehicle's current s-coordinate.
        #    - Handle the s-coordinate wraparound scenario using the module operation with `self.gb_max_s`.
        # 5. If both conditions are met, append the obstacle to the close_obs list.
        # 6. Return the close_obs list.

        # Only use obstacles that are within a threshold of the raceline, else we don't care about them
        obs_on_traj = [obs for obs in obstacles.obstacles if abs(obs.d_center) < self.obs_traj_tresh]

        # Only use obstacles that within self.lookahead in front of the car
        close_obs = []
        for obs in obs_on_traj:
            # Handle wraparound
            dist_in_front = (obs.s_center - frenet_state.pose.pose.position.x) % self.gb_max_s
            if dist_in_front < self.lookahead:
                close_obs.append(obs)
            else:
                # Not within lookahead
                pass
        return close_obs
    
    def _more_space(self, obstacle: Obstacle, gb_wpnts, gb_idx):
        """
        Determines the preferred direction (left or right) to overtake an obstacle based on available space.

        Parameters:
        - obstacle (Obstacle): The obstacle that needs to be overtaken.
        - gb_wpnts: Waypoints in the global plan.
        - gb_idx: Current index in the global waypoints list.
        
        Returns:
        - Tuple[str, float]: A tuple where the first element indicates the direction (e.g. "left" or "right") 
                             and the second element provides the apex distance from the raceline.
        """
        left_gap = abs(gb_wpnts[gb_idx].d_left - obstacle.d_left)
        right_gap = abs(gb_wpnts[gb_idx].d_right + obstacle.d_right)
        min_space = self.evasion_dist + self.spline_bound_mindist

        # 3:
        # 1. Check if both left_gap and right_gap are greater than min_space.
        # 2. If so, decide the best direction based on some criteria (shortest distance, safest, etc.).
        # 3. If only one gap is larger than min_space, decide the direction based on that gap.
        # 4. If neither gap is larger than min_space, determine a fallback strategy or indicate the inability to overtake.
        # 5. Calculate the value of d_apex based on the chosen direction.
        # 6. Ensure the return value is consistent with the function's documentation.

        if right_gap > min_space and left_gap < min_space:
            # Compute apex distance to the right of the opponent
            d_apex_right = obstacle.d_right - self.evasion_dist
            # If we overtake to the right of the opponent BUT the apex is to the left of the raceline, then we set the apex to 0
            if d_apex_right > 0:
                d_apex_right = 0
            return "right", d_apex_right

        elif left_gap > min_space and right_gap < min_space:
            # Compute apex distance to the left of the opponent
            d_apex_left = obstacle.d_left + self.evasion_dist
            # If we overtake to the left of the opponent BUT the apex is to the right of the raceline, then we set the apex to 0
            if d_apex_left < 0:
                d_apex_left = 0
            return "left", d_apex_left
        else:
            candidate_d_apex_left = obstacle.d_left + self.evasion_dist
            candidate_d_apex_right = obstacle.d_right - self.evasion_dist

            if abs(candidate_d_apex_left) <= abs(candidate_d_apex_right):
                # If we overtake to the left of the opponent BUT the apex is to the right of the raceline, then we set the apex to 0
                if candidate_d_apex_left < 0:
                    candidate_d_apex_left = 0
                return "left", candidate_d_apex_left
            else:
                # If we overtake to the right of the opponent BUT the apex is to the left of the raceline, then we set the apex to 0
                if candidate_d_apex_right > 0:
                    candidate_d_apex_right = 0
                return "right", candidate_d_apex_right

    def _do_spline(self, obstacles, gb_wpnts, frenet_state, wpnts):
        """
        Creates an evasion trajectory for the closest obstacle by splining between pre- and post-apex points.

        This function takes as input the obstacles to be evaded, and a list of global waypoints that describe a reference raceline.
        It only considers obstacles that are within a threshold of the raceline and generates an evasion trajectory for each of these obstacles.
        The evasion trajectory consists of a spline between pre- and post-apex points that are spaced apart from the obstacle.
        The spatial and velocity components of the spline are calculated using the `Spline` class, and the resulting waypoints and markers are returned.

        Args:
        - obstacles (ObstacleArray): An array of obstacle objects to be evaded.
        - gb_wpnts (WpntArray): A list of global waypoints that describe a reference raceline.
        - state (Odometry): The current state of the car.
        - wpnts (WpntArray): An array of waypoints that describe the evasion trajectory to the closest obstacle.

        Returns:
        - wpnts (WpntArray): An array of waypoints that describe the evasion trajectory to the closest obstacle.
        - mrks (MarkerArray): An array of markers that represent the waypoints in a visualization format.

        """
        # Return wpnts and markers
        mrks = MarkerArray()

        # Get spacing between wpnts for rough approximations
        wpnt_dist = gb_wpnts[1].s_m - gb_wpnts[0].s_m

        # Only use obstacles that are within a threshold of the raceline, else we don't care about them
        close_obs = self._obs_filtering(frenet_state=frenet_state, obstacles=obstacles)

        # If there are obstacles within the lookahead distance, then we need to generate an evasion trajectory considering the closest one
        if len(close_obs) > 0:
            # 2: Get the closest obstacle handling wraparound from the close_obs list
            closest_obs = min(close_obs, key=lambda obs: (obs.s_center - frenet_state.pose.pose.position.x) % self.gb_max_s)

            # Get Apex for evasion that is further away from the trackbounds
            s_apex = closest_obs.s_center
            # Get the global index of the apex
            gb_idx = int(s_apex / wpnt_dist) % self.gb_max_idx
            # Choose the correct side and compute the distance to the apex based on left of right of the obstacle
            more_space, d_apex = self._more_space(closest_obs, gb_wpnts, gb_idx)
            print('[Spliner] OT to the {} of the obstacle'.format(more_space))

            # Publish the point around which we are splining
            mrk = self.xy_to_point(x=gb_wpnts[gb_idx].x_m, y=gb_wpnts[gb_idx].y_m, opponent=False)
            self.closest_obs_pub.publish(mrk)

            # Choose wpnts from global trajectory for splining with velocity
            evasion_points = []
            spline_params = [i for i in range(-6, 8, 2)]
            for i, dst in enumerate(spline_params):
                si = s_apex + dst
                di = d_apex if dst == 0 else 0
                evasion_points.append([si, di])
            # Convert to nump
            evasion_points = np.array(evasion_points)

            # 4:
            # 1. Spatially spline the `d` dimension using `s` as the base:
            #    a. Define the spline resolution to be 0.1 m.
            #    b. Create the spatial spline using the evasion points.
            #    c. Generate the s-values (`evasion_s`) using numpy's `arange` based on the evasion points and the spline resolution.
            #    d. Calculate the corresponding d-values (`evasion_d`) by evaluating the spline at `evasion_s` values.

            spline_resolution = 0.1
            spatial_spline = Spline(x=evasion_points[:, 0], y=evasion_points[:, 1])
            evasion_s = np.arange(evasion_points[0, 0], evasion_points[-1, 0], spline_resolution)
            evasion_d = spatial_spline(evasion_s)


            # 4:
            # 2. Handle the wrapping of `s` values:
            #    a. Use the modulo operation to wrap `evasion_s` values around `self.gb_max_s`.

            evasion_s = evasion_s % self.gb_max_s

            # 4:
            # 3. Convert Frenet coordinates to Cartesian coordinates:
            #    a. Utilize the `converter` to obtain the Cartesian coordinates (`xy_arr`) from Frenet coordinates `evasion_s` and `evasion_d`.
            #    b. Optionally, create markers and waypoints if required by your program.

            xy_arr = self.converter.get_cartesian(evasion_s, evasion_d)

            for i in range(evasion_s.shape[0]):
                gb_wpnt_i = int((evasion_s[i] / wpnt_dist) % self.gb_max_idx)
                # Get V from gb wpnts
                vi = gb_wpnts[gb_wpnt_i].vx_mps 
                wpnts.wpnts.append(
                    self.xyv_to_wpnts(x=xy_arr[0, i], y=xy_arr[1, i], s=evasion_s[i], d=evasion_d[i], v=vi, wpnts=wpnts)
                )
                mrks.markers.append(self.xyv_to_markers(x=xy_arr[0, i], y=xy_arr[1, i], v=vi, mrks=mrks))

            # Fill the rest of OTWpnts
            wpnts.header.stamp = self.get_clock().now().to_msg()
            wpnts.header.frame_id = "map"
        return wpnts, mrks

    #########################################Visualization#############################################

    def xyv_to_markers(self, x, y, v, mrks: MarkerArray):
        mrk = Marker()
        mrk.header.frame_id = "map"
        mrk.header.stamp = self.get_clock().now().to_msg()
        mrk.type = mrk.CYLINDER
        mrk.scale.x = 0.1
        mrk.scale.y = 0.1
        mrk.scale.z = v / self.gb_vmax
        mrk.color.a = 1.0
        mrk.color.b = 0.75
        mrk.color.r = 0.75

        mrk.id = len(mrks.markers)
        mrk.pose.position.x = x
        mrk.pose.position.y = y
        mrk.pose.position.z = v / self.gb_vmax / 2
        mrk.pose.orientation.w = 1.0

        return mrk

    def xy_to_point(self, x, y, opponent=True):
        mrk = Marker()
        mrk.header.frame_id = "map"
        mrk.header.stamp = self.get_clock().now().to_msg()
        mrk.type = mrk.SPHERE
        mrk.scale.x = 0.5
        mrk.scale.y = 0.5
        mrk.scale.z = 0.5
        mrk.color.a = 0.8
        mrk.color.b = 0.65
        mrk.color.r = 1.0 if opponent else 0.0
        mrk.color.g = 0.65

        mrk.pose.position.x = x
        mrk.pose.position.y = y
        mrk.pose.position.z = 0.01
        mrk.pose.orientation.w = 1.0

        return mrk

    def xyv_to_wpnts(self, s, d, x, y, v, wpnts: WpntArray):
        wpnt = Wpnt()
        wpnt.id = len(wpnts.wpnts)
        wpnt.x_m = x
        wpnt.y_m = y
        wpnt.s_m = s
        wpnt.d_m = d
        wpnt.vx_mps = v
        return wpnt
    
def main():
    rclpy.init()
    node = ObstacleSpliner()
    rclpy.spin(node)
    rclpy.shutdown()




