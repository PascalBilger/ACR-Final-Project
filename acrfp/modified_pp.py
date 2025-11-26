#!/usr/bin/env python3

from rcl_interfaces.msg import ParameterDescriptor, ParameterValue
import rclpy
import numpy as np
from f110_msgs.msg import WpntArray
from nav_msgs.msg import Odometry

from lab5.visualization_pp import VizController as Controller

class ModPP(Controller):
    def __init__(self):
        super().__init__()
        
        self.otwaypoints_in_map = None 
        self.frenet_state = None
        
        self.create_subscription(Odometry, '/car_state/frenet/odom', self.state_cb, 10)
        self.create_subscription(WpntArray, '/planner/avoidance/otwpnts', self.otwpnts_cb, 10)
        
    def state_cb(self, msg: Odometry):
        self.frenet_state = msg

    def otwpnts_cb(self, msg: WpntArray):
        if len(msg.wpnts) > 0:
            self.otwaypoints_in_map = np.array([[wpnt.x_m, wpnt.y_m, wpnt.s_m] for wpnt in msg.wpnts])
        else:
            self.otwaypoints_in_map = None
            
    def check_if_otzone(self, s_pos, otwaypoints_in_map):
        """
        Checks if the current s position is within the Overtake Waypoints with wrapping

        Args:
            s_pos: current s position in frenet frame
            otwaypoints_in_map: Overtake Waypoints in map frame

        Returns:
            True if s_pos is within the Overtake Waypoints
        """
        # 5:
        # 1. Handle the wrapping scenario:
        #    a. If the start s-coordinate (i.e., the first entry) of `otwaypoints_in_map` is greater than the end s-coordinate (i.e., the last entry):
        #       - Check if `s_pos` is either greater than the start s-coordinate OR less than the end s-coordinate.
        #       - If either condition is met, return True.
        
        # 2. Handle the non-wrapping scenario:
        #    a. If `s_pos` lies between the start and end s-coordinates of `otwaypoints_in_map`, return True.
        
        # 3. If none of the above conditions are met, return False.
        
        if otwaypoints_in_map is not None:
            if otwaypoints_in_map[0, 2] > otwaypoints_in_map[-1, 2]:
                if s_pos > otwaypoints_in_map[0, 2] or s_pos < otwaypoints_in_map[-1, 2]:
                    return True
            elif s_pos > otwaypoints_in_map[0, 2] and s_pos < otwaypoints_in_map[-1, 2]:
                return True
        return False
    
    def find_pursuit_wpnt(self):
        """
        Overrides the method from the parent class
        Has to listen to new waypoints from the obstacle avoidance planner and select Overtake Waypoints if they are available

        Returns:
            waypoint as numpy array at a ceratin distance in front of the car
        """
        waypoints_in_map = self.waypoints_in_map
        # Check if our current s position of frenet state is within the Overtake Waypoints
        if self.otwaypoints_in_map is not None and self.frenet_state is not None:
            s = self.frenet_state.pose.pose.position.x
            if self.check_if_otzone(s_pos=s, otwaypoints_in_map=self.otwaypoints_in_map):
                waypoints_in_map = self.otwaypoints_in_map
                print('[Controller] We are in the Overtake Waypoints')

        # From here on PP is the same as in lab5!
        
        self.lookahead_distance = self.calc_lookahead_distance()

        # 1. find nearest waypoint idx to current position
        nearest_wpt_idx = np.argmin(np.linalg.norm(waypoints_in_map[:, :2] - self.position_in_map, axis = 1))

        # 2. find lookahead point, only looking in points forward
        looking_zone_len = 20
        if nearest_wpt_idx < len(waypoints_in_map)-looking_zone_len:
            idx_waypoint_at_distance = np.argmin(
                    np.abs(
                            np.linalg.norm(waypoints_in_map[nearest_wpt_idx:nearest_wpt_idx+looking_zone_len, :2] - self.position_in_map, axis = 1)-self.lookahead_distance
                        )
                )
            global_idx = nearest_wpt_idx+idx_waypoint_at_distance
        else:
            # consider wrapping!
            waypoints_search = np.concatenate([waypoints_in_map[nearest_wpt_idx:nearest_wpt_idx+looking_zone_len, :2], waypoints_in_map[:looking_zone_len, :2]])
            idx_waypoint_at_distance = np.argmin(
                    np.abs(
                            np.linalg.norm(waypoints_search - self.position_in_map, axis = 1)-self.lookahead_distance
                        )
                )
            global_idx = (nearest_wpt_idx+idx_waypoint_at_distance)%len(waypoints_in_map)
        return np.array(waypoints_in_map[global_idx]) 

def main():
    rclpy.init()
    mod_pp = ModPP()
    rclpy.spin(mod_pp)
    rclpy.shutdown()