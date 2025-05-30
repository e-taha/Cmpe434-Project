import time
import mujoco
import mujoco.viewer
import random
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from ProbabilisticRoadMap.probabilistic_road_map import prm_planning
from controller4 import PIDController
from controller5 import PurePursuitController
import cmpe434_dungeon as dungeon

import argparse


class CONTROLLER_TYPES:
    PP = 5
    PID = 4

class PLANNER_TYPES:
    PRM = 0
    RRT = 1
    RRT_D = 2
    RRT_STAR = 3
    RRT_STAR_D = 4

show_animation = False

shoot_range = 20.0
ray_count = 280
helper_geom_distance = 3  # [m]
distances_length = 121
show_helper_geoms = True
max_scoring_distance = 3.0
# Helper constructs for the viewer for pause/unpause functionality.
paused = False

max_reference_path_distance = 0.5  # [m]
back_min_steering = 0.4  # [rad]

kd = 1.0
ka = 1.5


def key_callback(keycode):
    global ka, kd
    print(f"Key pressed: {keycode}")
    if keycode == 265:  # Use up arrow key to increase ka
        ka += 0.1
    elif keycode == 264:  # Use down arrow key to decrease ka
        ka -= 0.1
    elif keycode == 262:  # Use right arrow key to increase kd
        kd += 0.1
    elif keycode == 263:  # Use left arrow key to decrease kd
        kd -= 0.1
    print(f"ka={ka}, kd={kd}")

# Pressing SPACE key toggles the paused state.
def mujoco_viewer_callback(keycode):
    global paused
    if keycode == ord(' '):  # Use ord(' ') for space key comparison
        paused = not paused

def quat_to_yaw(q):
    w, x, y, z = q
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    return np.arctan2(siny_cosp, cosy_cosp)

def angle_to_steering(angle):
    # Convert angle in radians to steering value
    return 16*angle

def calculate_velocity(angle, target_angle=0, max_velocity=1.0, min_velocity=0.1, distance=3.0):
    # Calculate the velocity according to the closeness to the best angle
    angle_difference = abs(angle - target_angle) / np.pi
    # difference = (1 - angle_difference)

    # The velocity function must be e^-angle_difference
    velocity = max_velocity * np.exp(-5*angle_difference)

    # If the distance is too small, decrease the velocity
    velocity = np.log(1 + distance*(np.e-1)/max_scoring_distance) * velocity

    if velocity < min_velocity:
        velocity = min_velocity
    return velocity

def score_angle(angle, distances,  best_angle=0.0, min_distance=0.1):
    global kd, ka
    n = len(distances)
    center = n // 2
    sigma = n / 6  # Standard deviation; adjust for sharpness
    x = np.arange(n)
    weights = np.exp(-0.5 * ((x - center) / sigma) ** 2)
    weights /= weights.sum()  # Normalize to sum to 1

    # Weighted sum of distances
    distance = np.sum(np.log(1 + distances - min_distance) * weights)
    if distance < min_distance:
        distance = min_distance
    response = kd * distance + ka * (1 - abs(angle - best_angle) / np.pi)
    if response is None or np.isnan(response) or np.isinf(response):
        print(f"Invalid score for angle {angle}: distance={distance}, best_angle={best_angle}, response={response}")
        response = 0.0
    return response

def find_best_angle(angles_and_distances, best_angle=0, min_distance=0.1, max_distance=3.0):
    """
    Score all angles based on the closeness to the best_angle and distance to the closest obstacle
    formula: score: kd*ln(1 + distance - min_distance) + ka(1 - abs(angle - best_angle)/pi)
    where distance is the distance to the closest obstacle, angle is the angle of the ray
    """
    scores = []
    # Consider each angles distance ant its neighboring angles'

    angles = angles_and_distances[0]
    distances = angles_and_distances[1]
    distances[distances < min_distance] = min_distance
    distances[distances > max_distance] = max_distance

    # Start scoring from distances_length//2 to avoid index out of range
    for i in range(distances_length//2, len(angles) - distances_length//2):
        angle = angles[i]
        distances_subset = distances[i - distances_length//2:i + distances_length//2 + 1]
        # Calculate the score for the angle
        score = score_angle(angle, distances_subset, best_angle, min_distance)
        scores.append(score)
    best_index = np.argmax(scores)

    best_index += distances_length // 2  # Adjust index to match the original angles array
    if best_index < 0 or best_index >= len(angles):
        print(f"Best index {best_index} out of range for angles array of length {len(angles)}")
        raise IndexError("Best index out of range")

    best_angle = angles_and_distances[0][best_index]
    best_distance = angles_and_distances[1][best_index]
    # print(f"Best angle: {best_angle}, Best distance: {best_distance}, Best index: {best_index}, Score: {scores[best_index - distances_length//2]}")
    return best_angle, best_distance, best_index


def main():

    # Uncomment to start with an empty model
    # scene_spec = mujoco.MjSpec() 
    parser = argparse.ArgumentParser()
    parser.add_argument("--controller", type=int, default=CONTROLLER_TYPES.PP)
    parser.add_argument("--velocity", type=float, default=5)
    parser.add_argument("--planner", type=int, default=PLANNER_TYPES.PRM)

    args = parser.parse_args()
    controller_type = args.controller
    planner_type = args.planner
    # Uncomment to start with an empty model
    # Add walls to the scene: Walls are rectangular prisms with a specified position and size.
    # We need to define the walls in here
    global_counter = 0

    obstacles = []
    extended_obstacles = []

    def old_add_wall_obstacle(x_s,y_s,x_e,y_e,step_size = 0.5):
        x1 = x_s
        y1 = y_s
        x2 = x_e
        y2 = y_e
        print(f"Adding wall obstacle from ({x1}, {y1}) to ({x2}, {y2})")
        if x1 == x2:
            y = y1
            if y2 > y1:
                while y <= y2:
                    extended_obstacles.append((x1, y))
                    y += step_size
            else:
                while y >= y2:
                    extended_obstacles.append((x1, y))
                    y -= step_size

        else:
            x = x1
            if x2 > x1:
                while x <= x2:
                    extended_obstacles.append((x, y1))
                    x += step_size
            else:
                while x >= x2:
                    extended_obstacles.append((x, y1))
                    x -= step_size

    def add_wall_obstacle(x_1, y_1, x_2, y_2):
        obstacles.append((x_1, y_1, x_2, y_2))
        old_add_wall_obstacle(x_1, y_1, x_2, y_2)
        print(f"Adding wall obstacle at ({x_1}, {y_1}) to ({x_2}, {y_2})")

    # Load existing XML models
    scene_spec = mujoco.MjSpec.from_file("../scenes/empty_floor.xml")

    tiles, rooms, connections = dungeon.generate(3, 2, 8)
    for index, r in enumerate(rooms):
        (xmin, ymin, xmax, ymax) = dungeon.find_room_corners(r)
        scene_spec.worldbody.add_geom(name='R{}'.format(index), type=mujoco.mjtGeom.mjGEOM_PLANE, size=[(xmax-xmin)+1, (ymax-ymin)+1, 0.3], rgba=[0.8, 0.6, 0.4, 1],  pos=[(xmin+xmax), (ymin+ymax), 0.005])

    for pos, tile in tiles.items():
        if tile == "#":
            add_wall_obstacle(2*pos[0] - 1, 2*pos[1] - 1, 2*pos[0] + 1, 2*pos[1] - 1)
            add_wall_obstacle(2*pos[0] - 1, 2*pos[1] + 1, 2*pos[0] + 1, 2*pos[1] + 1)
            add_wall_obstacle(2*pos[0] - 1, 2*pos[1] - 1, 2*pos[0] - 1, 2*pos[1] + 1)
            add_wall_obstacle(2*pos[0] + 1, 2*pos[1] - 1, 2*pos[0] + 1, 2*pos[1] + 1)
            scene_spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[1, 1, 0.1], rgba=[0.8, 0.6, 0.4, 1],  pos=[pos[0]*2, pos[1]*2, 0])

    start_pos = random.choice([key for key in tiles.keys() if tiles[key] == "."])
    final_pos = random.choice([key for key in tiles.keys() if tiles[key] == "." and key != start_pos])

    scene_spec.worldbody.add_site(name='start', type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.5, 0.5, 0.01], rgba=[0, 0, 1, 1],  pos=[start_pos[0]*2, start_pos[1]*2, 0])
    scene_spec.worldbody.add_site(name='finish', type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.5, 0.5, 0.01], rgba=[1, 0, 0, 1],  pos=[final_pos[0]*2, final_pos[1]*2, 0])


    robot_spec = mujoco.MjSpec.from_file("../models/mushr_car/model.xml")

    # Add robots to the scene:
    # - There must be a frame or site in the scene model to attach the robot to.
    # - A prefix is required if we add multiple robots using the same model.
    scene_spec.attach(robot_spec, frame="world", prefix="robot-")
    scene_spec.body("robot-buddy").pos[0] = start_pos[0] * 2
    scene_spec.body("robot-buddy").pos[1] = start_pos[1] * 2

    # Randomize initial orientation
    yaw = np.random.uniform(-np.pi, np.pi)
    euler = np.array([0.0, 0.0, yaw], dtype=np.float64)
    quat = np.zeros(4, dtype=np.float64)
    mujoco.mju_euler2Quat(quat, euler, 'xyz')
    scene_spec.body("robot-buddy").quat[:] = quat

    reference_path = []
    if planner_type == PLANNER_TYPES.PRM:
        ox = [x[0] for x in extended_obstacles]
        oy = [y[1] for y in extended_obstacles]
        sx = start_pos[0] * 2  # [m]
        sy = start_pos[1] * 2  # [m]
        gx = final_pos[0] * 2  # [m]
        gy = final_pos[1] * 2  # [m]
        if show_animation:
            plt.plot(ox, oy, ".k")
            plt.plot(sx, sy, "^r")
            plt.plot(gx, gy, "^c")
            plt.grid(True)
            plt.axis("equal")

        r_p_x, r_p_y = prm_planning(sx, sy, gx, gy, ox, oy, 0.5)
        reference_path = list(zip(r_p_x, r_p_y))

        if show_animation:
            plt.plot(r_p_x, r_p_y, "-r")
            plt.pause(0.001)
            plt.show()
        
    reference_path = reference_path[::-1]

    # Increase the number of points in the reference path
    if len(reference_path) > 1:
        new_reference_path = []
        for i in range(len(reference_path) - 1):
            start = np.array(reference_path[i])
            end = np.array(reference_path[i + 1])
            num_points = int(np.linalg.norm(end - start) / max_reference_path_distance)
            if num_points < 2:
                new_reference_path.append(start)
                continue
            for j in range(num_points):
                point = start + (end - start) * (j / (num_points - 1))
                new_reference_path.append(point)
        reference_path = new_reference_path

    # Remove concurrent duplicate points in the reference path
    reference_path = [reference_path[i] for i in range(len(reference_path)) if i == 0 or not np.allclose(reference_path[i], reference_path[i - 1])]

    # If the last element of the reference path is not the final position, add it
    if len(reference_path) > 0 and not np.allclose(reference_path[-1], [final_pos[0] * 2, final_pos[1] * 2]):
        reference_path.append([final_pos[0] * 2, final_pos[1] * 2])
    # for i in range(len(reference_path)):
        # scene_spec.worldbody.add_body(
        #     pos=[reference_path[i][0], reference_path[i][1], 0],
        #     quat=[0, 0, 0, 0],
        # ).add_geom(
        #     type=mujoco.mjtGeom.mjGEOM_BOX,
        #     size=[0.03, 0.03, 0.01],
        #     rgba=[1, 0, 0, 1],
        #     # Make the geom non-collidable
        #     contype=0,
        #     conaffinity=0
        # )

    # Add obstacles to the scene
    for i, room in enumerate(rooms):
        obs_pos = random.choice([tile for tile in room if tile != start_pos and tile != final_pos])
        scene_spec.worldbody.add_geom(
            name='Z{}'.format(i), 
            type=mujoco.mjtGeom.mjGEOM_CYLINDER, 
            size=[0.2, 0.05, 0.1], 
            rgba=[0.8, 0.0, 0.1, 1],  
            pos=[obs_pos[0]*2, obs_pos[1]*2, 0.08]
        )

    # Initalize our simulation
    # Roughly, m keeps static (model) information, and d keeps dynamic (state) information. 
    m = scene_spec.compile()
    d = mujoco.MjData(m)

    obstacles = [m.geom(i).id for i in range(m.ngeom) if m.geom(i).name.startswith("Z")]
    uniform_direction_dist = sp.stats.uniform_direction(2)
    obstacle_direction = [[x, y, 0] for x,y in uniform_direction_dist.rvs(len(obstacles))]
    unused = np.zeros(1, dtype=np.int32)
    if show_animation:
        # Plot the extended obstacles
        plt.figure()
        for x, y in extended_obstacles:
            plt.plot(x, y, 'ro')
        # plt.xlim(-10, 10)
        # plt.ylim(-10, 10)
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.title('Extended Obstacles')
        plt.grid()
        plt.show()
        return
    
    with mujoco.viewer.launch_passive(m, d, key_callback=key_callback) as viewer:


        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        viewer.cam.fixedcamid = m.camera("robot-third_person").id

        # These actuator names are defined in the model XML file for the robot.
        # Prefixes distinguish from other actuators from the same model.
        velocity = d.actuator("robot-throttle_velocity")
        steering = d.actuator("robot-steering")
        PID_controller = PIDController(20, 0.01, 5)
        PP_Controller = PurePursuitController(1,0.2965)

        if controller_type == CONTROLLER_TYPES.PP:
            target_geom_index = viewer.user_scn.ngeom
            mujoco.mjv_initGeom(
                viewer.user_scn.geoms[target_geom_index],
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=[0.05, 0, 0],
                pos=np.array([0, 0, -0.005]),
                mat=np.eye(3).flatten(),
                rgba=[1, 1, 1, 1],
            )
            viewer.user_scn.ngeom += 1
        
        #  Add helper geoms to see the possible courses

        ray_geom_indexes = []
        if show_helper_geoms:
            for i in range(ray_count):
                geom_index = viewer.user_scn.ngeom
                mujoco.mjv_initGeom(
                    viewer.user_scn.geoms[geom_index],
                    type=mujoco.mjtGeom.mjGEOM_SPHERE,
                    size=[0.05, 0, 0],
                    pos=np.array([0, 0, -0.005]),
                    mat=np.eye(3).flatten(),
                    rgba=[0, 0, 0, 1],
                )
                viewer.user_scn.ngeom += 1
                ray_geom_indexes.append(geom_index)
            
            #  Add another helper geom to see the best angle course
            best_angle_geom_index = viewer.user_scn.ngeom
            mujoco.mjv_initGeom(
                viewer.user_scn.geoms[best_angle_geom_index],
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=[0.05, 0, 0],
                pos=np.array([0, 0, -0.005]),
                mat=np.eye(3).flatten(),
                rgba=[1, 0, 0, 1],
            )
            viewer.user_scn.ngeom += 1

        # Close the viewer automatically after 30 wall-clock-seconds.
        start = time.time()
        back_mode = False
        step_start = time.time()
        def get_error(x, y):
            # Get the closest point on the reference path
            # Find the closest point on the reference path
            car_position = np.array([x, y])
            distances = np.linalg.norm(reference_path - car_position, axis=1)
            closest_index = np.argmin(distances)
            closest_point = reference_path[closest_index]

            # Cross-track error (distance to the closest point)
            cross_track_error = distances[closest_index]

            # The sign of the cross-track error depends on the side of the car
            # (left or right) with respect to the reference path
            previous_point = None
            next_point = None
            if closest_index == 0:
                next_point = np.array(reference_path[closest_index + 1])
                previous_point = np.array(reference_path[-1])
            elif closest_index == len(reference_path) - 1:
                next_point = np.array(reference_path[0])
                previous_point = np.array(reference_path[closest_index - 1])
            else:
                previous_point = np.array(reference_path[closest_index - 1])
                next_point = np.array(reference_path[closest_index + 1])
            v1 = car_position - previous_point
            v2 = next_point - previous_point
            cross_product = np.cross(v1, v2)
            cross_track_error = np.sign(cross_product) * cross_track_error


            # Assuming reference_path is a sequence of (x, y) points
            if closest_index < len(reference_path) - 1:
                next_point = reference_path[closest_index + 1]
                path_heading = np.arctan2(next_point[1] - closest_point[1], next_point[0] - closest_point[0])
                car_heading = np.arctan2(car_position[1] - closest_point[1], car_position[0] - closest_point[0])
                heading_error = path_heading - car_heading
            else:
                heading_error = 0

            return cross_track_error


        def step():
            nonlocal back_mode
            for i, x in enumerate(obstacles):
                dx = obstacle_direction[i][0]
                dy = obstacle_direction[i][1]

                px = m.geom_pos[x][0]
                py = m.geom_pos[x][1]
                pz = 0.02

                nearest_dist = mujoco.mj_ray(m, d, [px, py, pz], obstacle_direction[i], None, 1, -1, unused)

                if nearest_dist >= 0 and nearest_dist < 0.4:
                    obstacle_direction[i][0] = -dy
                    obstacle_direction[i][1] = dx

                m.geom_pos[x][0] = m.geom_pos[x][0]+dx*0.001
                m.geom_pos[x][1] = m.geom_pos[x][1]+dy*0.001

            steering_value = 0
            steering_angle = 0
            nonlocal step_counter
            car_position = d.xpos[d.model.body("robot-buddy").id]
            x = car_position[0]
            y = car_position[1]
            isEnd = False
            if controller_type == CONTROLLER_TYPES.PID:
                error = get_error(x, y)
                steering_value = PID_controller.update(error=error)
            elif controller_type == CONTROLLER_TYPES.PP:
                target_point, isEnd = PP_Controller.find_lookahead_point(np.array([x, y]), reference_path)
                # Update the target geom position
                viewer.user_scn.geoms[target_geom_index].pos = np.array([target_point[0], target_point[1], -0.005])
                # Get car's quaternion
                car_quat = d.xquat[d.model.body("robot-buddy").id]
                # Calculate yaw angle from quaternion
                # For a quaternion [w, x, y, z], yaw = atan2(2(wz + xy), 1 - 2(y^2 + z^2))
                w, xc, yc, z = car_quat
                car_direction = np.arctan2(2 * (w * z + xc * yc), 1 - 2 * (yc * yc + z * z))
                steering_angle = PP_Controller.calculate_steering_angle(car_position, target_point, car_direction)
            max_steering = 10
            min_steering = -10

            if not viewer.is_running():
                return

            # if(step_counter % 20 == 0):
            #     i = viewer.user_scn.ngeom
            #     mujoco.mjv_initGeom(
            #             viewer.user_scn.geoms[i],
            #             type=mujoco.mjtGeom.mjGEOM_SPHERE,
            #             size=[0.03, 0, 0],
            #             pos=np.array([x, y, -0.005]),
            #             mat=np.eye(3).flatten(),
            #             rgba=[0, 1, 0, 1],
            #     )
            #     viewer.user_scn.ngeom += 1
            step_counter += 1

            car_buddy = m.body("robot-buddy")
            car_position_raw = d.xpos[d.model.body("robot-buddy").id]
            car_pos = [car_position_raw[0], car_position_raw[1], 0.1]
            yaw = quat_to_yaw(d.xquat[car_buddy.id])
            ray_dir = [np.cos(yaw), np.sin(yaw), 0]
            # print("Yaw: {}".format(yaw))
            offset = -0.6
            ray_start = [
                car_pos[0] + ray_dir[0] * offset,
                car_pos[1] + ray_dir[1] * offset,
                car_pos[2]
            ]

            # nearest_dist = mujoco.mj_ray(m, d, ray_start, ray_dir, None, 1, -1, unused)
            # Shoot range by percentage

            first_yaw = yaw + (shoot_range/100) * np.pi;
            ray_dirs = []
            ray_distance = (shoot_range / 100) * 2 * np.pi / (ray_count - 1)
            angles = np.array([], dtype=np.float64)
            for i in range(0, ray_count):
                ray_yaw = first_yaw - ray_distance * i
                ray_dirs.append(np.cos(ray_yaw))
                ray_dirs.append(np.sin(ray_yaw))
                ray_dirs.append(0.0)
                angles = np.append(angles, ray_yaw - yaw)

                # Update helper geom positions
                if show_helper_geoms:
                    geom_index = ray_geom_indexes[i]
                    viewer.user_scn.geoms[geom_index].pos = np.array([
                        ray_start[0] + ray_dirs[-3] * helper_geom_distance,
                        ray_start[1] + ray_dirs[-2] * helper_geom_distance,
                        ray_start[2]
                    ])
                    viewer.user_scn.geoms[geom_index].rgba = [0, 0, 0, 1]  # Reset color to black

            # Add steering angle from controller to the angles array and ray_dirs
            # angles = np.append(angles, steering_angle)
            # ray_dirs.append(np.cos(steering_angle + yaw))
            # ray_dirs.append(np.sin(steering_angle + yaw))
            # ray_dirs.append(0.0)

            if show_helper_geoms:
                # Update the best angle helper geom position and color
                viewer.user_scn.geoms[best_angle_geom_index].pos = np.array([
                    ray_start[0] + np.cos(steering_angle + yaw) * helper_geom_distance,
                    ray_start[1] + np.sin(steering_angle + yaw) * helper_geom_distance,
                    ray_start[2]
                ])
                viewer.user_scn.geoms[best_angle_geom_index].rgba = [1, 0, 0, 1]  # Reset color to red


            # print("Ray angles:", angles)
            # print("Steering angle:", steering_angle)
            nray = ray_count #+ 1  # +1 for the steering angle ray

            # Prepare output arrays
            geomid = np.full(nray, -1, dtype=np.int32)         # Output: geom IDs hit by each ray
            dist = np.full(nray, -1.0, dtype=np.float64)       # Output: distances for each ray
            geomgroup = np.ones(6, dtype=np.uint8)
            geomgroup[1] = 0  # Set group 1 (car) to 0 to exclude
            geomgroup[2] = 0

            mujoco.mj_multiRay(m, d, ray_start, ray_dirs, geomgroup, 1, -1, geomid, dist, nray, max_scoring_distance - offset)

            # replace all -1 distances with max_scoring_distance - offset
            dist[dist < 0] = max_scoring_distance - offset
            # subtract the offset from the distances
            dist += offset
            # print("Distances:", dist)

            angles_and_distances = np.array([angles, dist])
            best_angle, best_distance, best_index = find_best_angle(angles_and_distances, steering_angle, max_distance=(max_scoring_distance if not isEnd else 1.0))
            if isEnd:
                pass
                # print("ISENDISENDISENDISEND")
            # print("Best angle: {}, Best distance: {}".format(best_angle, best_distance))

            if best_distance < 0.7:
                back_mode = True
            elif best_distance > 2:
                back_mode = False
                

            if show_helper_geoms:
                # Update the best angle helper geom color as green
                if best_index < len(ray_geom_indexes):
                    # set the distances length range geoms's color to grey
                    for i in range(best_index - distances_length//2, best_index + distances_length//2 + 1):
                        if i < 0 or i >= len(ray_geom_indexes):
                            continue
                        viewer.user_scn.geoms[ray_geom_indexes[i]].rgba = [0.5, 0.5, 0.5, 1]
                    viewer.user_scn.geoms[ray_geom_indexes[best_index]].rgba = [0, 1, 0, 1]
                else:
                    viewer.user_scn.geoms[best_angle_geom_index].rgba = [0, 1, 0, 1]
            
            
            velocity.ctrl = calculate_velocity(best_angle, steering_angle, args.velocity, 0.1, best_distance)


            steering_value = angle_to_steering(best_angle)

            if steering_value > max_steering:
                steering_value = max_steering
            elif steering_value < min_steering:
                steering_value = min_steering
            
            steering.ctrl = steering_value

            velocity.ctrl = velocity.ctrl * (1  - abs(steering_value) / max_steering)
            if velocity.ctrl < 0.1:
                velocity.ctrl = 0.1

            if back_mode:
                # If back mode is on, reverse the steering angle
                velocity.ctrl = -velocity.ctrl
                if velocity.ctrl > -0.4:
                    velocity.ctrl = -0.4
                # add the supposed steering angle to the steering value
                steering.ctrl = angle_to_steering(-steering_angle)

                if abs(steering.ctrl) < back_min_steering:
                    steering.ctrl = back_min_steering if steering.ctrl > 0 else -back_min_steering

            mujoco.mj_step(m, d)
            viewer.sync()

  

        while viewer.is_running() and time.time() - start < 3000:
            step_counter = 0

            step_start = time.time()
            car_velocity = d.sensor("robot-velocimeter").data[0]
            car_acceleration = d.sensor("robot-accelerometer").data[0]

            if not paused:
                time_until_next_step = m.opt.timestep - (time.time() - step_start)
                start_time = d.time
                while viewer.is_running() and d.time - start_time < 30:
                    step()


if __name__ == "__main__":
    main()
