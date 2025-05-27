import math
import numpy as np

class PurePursuitController:
    def __init__(self, lookahead_distance, wheelbase):
        self.lookahead_distance = lookahead_distance
        self.wheelbase = wheelbase

    def calculate_steering_angle(self, current_pose, target_pose, current_direction):
        # Calculate the steering angle using the pure pursuit method
        # This is a placeholder implementation and should be replaced with the actual calculation
        # Calculate the distance to the target
        dx = target_pose[0] - current_pose[0]
        dy = target_pose[1] - current_pose[1]

        # Calculate the heading angle error
        heading_angle = math.atan2(dy, dx)
        heading_error = heading_angle - current_direction

        # Calculate the curvature
        curvature = 2 * math.sin(heading_error) / self.lookahead_distance

        # Calculate the steering angle
        steering_angle = math.atan(curvature * self.wheelbase)

        return steering_angle
    
    def find_lookahead_point(self, current_pose, reference_path):

        # Find the closest point on the reference path first
        isEnd = False
        distances = np.linalg.norm(reference_path - current_pose, axis=1)
        closest_index = np.argmin(distances)
        closest_point = reference_path[closest_index]

        # Find the lookahead point
        lookahead_index = closest_index + 1

        while lookahead_index != closest_index:
            if lookahead_index >= len(reference_path):
                lookahead_index -= 1
                isEnd = True
                break
            # Check if the lookahead point is too far away
            if distances[lookahead_index] > self.lookahead_distance:
                break
            lookahead_index += 1

        previous_index = lookahead_index - 1
        if previous_index < 0:
            previous_index = len(reference_path) - 1

        if distances[closest_index] > self.lookahead_distance:
            previous_index = closest_index 
            lookahead_index = closest_index + 1
            if lookahead_index >= len(reference_path):
                lookahead_index = 0

        # Calculate the intersection point of the line segment from previous point to lookahead point if it is not the end of the path
        if isEnd:
            return reference_path[-1], isEnd
        x1, y1 = reference_path[previous_index]
        x2, y2 = reference_path[lookahead_index]
        xc, yc = current_pose

        A = (x2 - x1) ** 2 + (y2 - y1) ** 2
        B = 2 * ((x2 - x1) * (x1 - xc) + (y2 - y1) * (y1 - yc))
        C = (x1 - xc) ** 2 + (y1 - yc) ** 2 - self.lookahead_distance ** 2

        discriminant = B ** 2 - 4 * A * C

        if discriminant < 0:
            return reference_path[lookahead_index], isEnd
        
        t1 = (-B + math.sqrt(discriminant)) / (2 * A)
        t2 = (-B - math.sqrt(discriminant)) / (2 * A)

        t = t1
        if t < 0 or t > 1:
            t = t2
            if t < 0 or t > 1:
                return reference_path[lookahead_index], isEnd
            
        x = x1 + t * (x2 - x1)
        y = y1 + t * (y2 - y1)

        return np.array([x, y]), isEnd

        
        

        
        

    
    def calculate_velocity(self, current_pose, target_pose, current_velocity):
        # Calculate the velocity using the pure pursuit method
        # This is a placeholder implementation and should be replaced with the actual calculation
        pass

    
