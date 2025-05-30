(Introductory part for the autonomous driving)

In this project, we are going to create a simulation where we can test the autonomous driving capabilities of a car. The main scenario we will focus on is this: A car tries to reach a goal point from a starting point while avoiding obstacles inside an environment of labyrinth-like connected rooms. The car will be able to sense its environment and make decisions based on the information it receives.

There are 3 main kinds of algorithms regarding the movement of the car:

1. **Path Planning Algorithms**: These algorithms are responsible for determining the best path from the starting point to the goal point while avoiding obstacles. Examples include A\* algorithm, Dijkstra's algorithm, and Rapidly-exploring Random Trees (RRT).
2. **Path Tracking Algorithms**: Once a path is determined, these algorithms help the car follow the path accurately. They adjust the car's steering and speed to stay on course. Examples include Pure Pursuit and Stanley controller.
3. **Obstacle Avoidance Algorithms**: These algorithms help the car detect and avoid obstacles in real-time while driving. Examples include Dynamic Window Approach (DWA) and Vector Field Histogram (VFH).

In my previous simulations, I got the best result with Probabilistic Roadmap (PRM) regarding path planning. So, I decided to use PRM for this project as well. For path tracking, I will use the Pure Pursuit algorithm, which is simple and effective for following a path. And I wanted to use a much more less used controller which is compared to PID. There are so many papers and research on PID, but I got a very good result with PP controller and since I wanted to use DWA for obstacle avoidance, I thought their working principles are similar, since both of them are based on the idea of predicting the future state of the car and adjusting the control inputs accordingly.

Unfortunately, I use MacOs on my computer and MacOs has a habit of not supporting generally anything regarding robotics and simulation tools, which makes it challenging to implement and test certain algorithms. To run Mujoco with python on MacOs, I need to use mjpython instead of python3, which has some compatibility issues regarding threads with some packages such as matplotlib. To see the generated path of PRM on matplotlib and make sure that it works, I decided to first generate and see the path without running mujoco, then implement the other things. This way, I can ensure that the path planning works correctly before integrating it into the simulation environment.

# Path Planning with PRM

In my previous simulations, PRM was the only algorithm that finds the path nearly all the time, so I decided to use it again. PRM is a sampling-based algorithm that builds a roadmap of the environment by randomly sampling points and connecting them if they are reachable. In my simulations, I initially set the number of sampled points to 500, but I found that it was not enough to cover the environment adequately and sometimes (nearly %30 - 40% of the time), it could not find a path. So, I increased the number of sampled points to 5000, which significantly improved the pathfinding capability of PRM, but it was taking so much time.My KNN (k-nearest neighbors) number was set to 10 and max edge length was set to 30. I thought to change them too, but when I set the number of sampled points to 1000, it worked well enough, so I didn't change them. After generating the path, I visualized it using matplotlib to ensure that the path was generated correctly.

![PRM Path](https://github.com/user-attachments/assets/5006fbe5-f1b3-4f47-90c5-7da405172bd8)
![PRM Path 2](https://github.com/user-attachments/assets/83ac87da-c05b-4682-b78d-2c55a3174822)
![PRM Path 3](https://github.com/user-attachments/assets/b2f07362-5686-4f54-9e80-decf86a78416)


# Path Tracking with Pure Pursuit

Pure Pursuit is a simple and effective path tracking algorithm that uses a lookahead distance to determine the target point on the path. The car then steers towards this target point to follow the path. I implemented the Pure Pursuit algorithm in Python and tested it with the generated PRM path. The algorithm works by calculating the curvature of the path at the target point and adjusting the steering angle accordingly. Since the steering value doesn't have a general unit, I product the steering angle in radians with 16 and limited it between -10 and 10. I also implemented a simple speed control mechanism that adjusts the speed based on the steering angle to ensure smooth driving. The car's speed is reduced when the steering angle is large, and it is increased when the steering angle is small.

# Obstacle Avoidance with DWA

Dynamic Window Approach (DWA) is an obstacle avoidance algorithm that uses the car's dynamic constraints to determine the best velocity and steering angle to avoid obstacles while following the path. DWA considers the car's current state, the target path, and the obstacles in the environment to compute a set of feasible velocities and steering angles. It then evaluates these candidates based on a cost function that considers factors such as distance to the goal, distance to obstacles, and smoothness of the trajectory. Although DWA considers different set of velocities and steering angles, I decided to use just the steering angle from DWA and calculating the speed separately based on the steering angle and the distance. Since I have 2 different algorithms that effects the steering angle, I needed to combine them. One solution I thought was using DWA if there is an obstacle in the environment, otherwise using Pure Pursuit. But this kind of approach doesn't provide a smooth transition between the two algorithms, which can lead to abrupt changes in the car's behavior. Instead, I decided to use a scoring function that combines the outputs of both algorithms. The scoring function evaluates the steering angle from Pure Pursuit and DWA, and selects the one with the higher score. This way, I can ensure that the car follows the path while avoiding obstacles effectively.

To find distances to the obstacles, I used mujoco's built-in mj_multiRay function, which allows me to cast rays from a defined position in different directions and check for collisions with obstacles. For regarding the multiray function, I created a set of constants that will help me to use the function more easily. These are:

```python
shoot_range = 25.0 # The percentage of the circle that the rays will cover
ray_count = 20 # The number of rays to cast
max_scoring_distance = 3.0 # The maximum distance to consider for scoring
```

These constants define the range of the rays, the number of rays to cast, and the maximum distance to consider for scoring. The `shoot_range` is set to 25.0, which means that the rays will cover 25% of the circle around the car. The `ray_count` is set to 20, which means that 20 rays will be cast in total. The `max_scoring_distance` is set to 3.0, which means that only obstacles within this distance will be considered for scoring.

I was casting these rays but these angles were not very continuous, which means when the selected angle changes, the steering angle also changes abruptly and doesn't offer a smooth transition. Since the main reason I cast these rays were the DWA, I decided to cast another ray which its angle is the same as the steering angle from Pure Pursuit. This way, I can ensure that the car avoids obstacles while following the path smoothly. By that, I get "ray_count + 1" angle and distance pairs in total.

```python
mujoco.mj_multiRay(m, d, ray_start, ray_dirs, None, 1, -1, geomid, dist, nray, max_scoring_distance)
```

The `mujoco.mj_multiRay` function is used to cast rays from the car's position in different directions. The parameters are as follows:

- `m`: The Mujoco model.
- `d`: The Mujoco data structure that contains the current state of the simulation.
- `ray_start`: The starting position of the rays, which is the car's position in this case.
- `ray_dirs`: The directions of the rays, which are calculated based on the car's orientation and the angles defined by `shoot_range` and `ray_count`.
- `None`: This parameter is used when there are some geom groups that we want to ignore. In this case, we don't have any geom groups to ignore, so we set it to `None`.
- `geomid`: The array that if a collision occurs, the index of the geom that was hit will be stored in this array.
- `dist`: The array that will store the distances to the obstacles hit by the rays.
- `nray`: The number of rays to cast, which is equal to `ray_count + 1`.
- `max_scoring_distance`: The cutoff distance for raycasting. Since the distances bigger than this value will not be considered for scoring, we set it to `max_scoring_distance`.

The results can contain `-1` values, which means that the ray didn't hit any obstacle. In this case, I set the distance to `max_scoring_distance`, which means that the ray didn't hit any obstacle within the scoring distance.

# Scoring Function

There are two important factors that I considered for scoring each angle distance pairs: the angle difference to the goal and the distance to the obstacles. First, I considered a linear weighted sum of these two factors, where the angle difference to the goal is multiplied by a weight factor and the distance to the obstacles is multiplied by another weight factor.

I defined the scoring function as follows:

```python
def score_angle(angle, distance,  best_angle=0.0, min_distance=0.1):

    kd = 1.0
    ka = 1.5
    if distance < min_distance:
        distance = min_distance
    response = kd * (distance - min_distance) + ka * (1 - abs(angle - best_angle) / np.pi)
    return response
```

This function takes the angle, distances to obstacles, the best angle (which is the steering angle from Pure Pursuit), and a minimum distance value as inputs. It calculates the score based on the distance to the closest obstacle and the angle difference to the best angle. The `kd` and `ka` are weight factors for distance and angle, respectively. The function returns a response value that represents the score for the given angle.

It worked well, but it didn't felt quite right to add distance value linearly. For example, if there are 2 pairs which one distance is 2.8 and other distance is 2.9, the difference between them is 0.1, which is not a big difference. But if there are 2 pairs which one distance is 0.8 and other distance is 0.9, the difference between them is 0.1 again, but this time it is a significant difference. Which I concluded that it is best to use a function whose partial derivative according to the distance is decreasing. So, I decided to use a logarithmic function for the distance. The scoring function is defined as follows:

```python
def score_angle(angle, distance,  best_angle=0.0, min_distance=0.1):

    kd = 1.0
    ka = 1.5
    if distance < min_distance:
        distance = min_distance
    response = kd * np.log(1 + distance - min_distance) + ka * (1 - abs(angle - best_angle) / np.pi)
    return response
```

This function uses the logarithmic function to calculate the score based on the distance to the closest obstacle. The `np.log(1 + distance - min_distance)` part ensures that the score increases logarithmically with the distance. In my first try, it worked surprisingly well. So much so that I didn't change the function formula on most of the project. I changed the weight factors to get better results. Which they were initially equal to 1. I changed ka to 1.5, which worked well for my simulation. I also increased the number of rays to 280 and decreased the range from 25.0 to 20.0 and it worked much better. But there were still some cases where the car was not able to avoid obstacles effectively, and the reason for that is it doesn't consider the width of the car, so it generally targets the edge of the obstacle, which can lead to collisions. To solve this problem, I decided to include the ray distances that are close to the steering angle in scoring and took the minimum of these distances. This way, I can ensure that the car avoids obstacles effectively while following the path.

```python
def score_angle(angle, distances, best_angle=0.0, min_distance=0.1):
    kd = 1.0
    ka = 1.5
    if min(distances) < min_distance:
        distances = [max(d, min_distance) for d in distances]
    response = kd * np.log(1 + min(distances) - min_distance) + ka * (1 - abs(angle - best_angle) / np.pi)
    return response
```

I set a constant called `distances_length` and set it to 61. This constant defines the number of distances to consider for scoring. The `distances` parameter is a list of distances to the obstacles hit by the rays, and I take the minimum of these distances to calculate the score. This way, I can ensure that the car avoids obstacles effectively while following the path. But there was still a problem. When an obstacle enters the ray casting area, the car changes its steering angle abruptly. To overcome this, instead of taking the minimum of the distances, I decided to take the weighted sum of the distances that are close to the steering angle. The weight factors becomes much higher for the distances that are close to the steering angle, and lower for the distances that are far from the steering angle. To reach this, I wanted my weight factors to form a Gaussian distribution around the steering angle. My final scoring function was this:

```python
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
```

This function uses a Gaussian distribution to calculate the weights for the distances, which ensures that the distances that are close to the steering angle have higher weights, and the distances that are far from the steering angle have lower weights. The `sigma` parameter defines the standard deviation of the Gaussian distribution, which controls how sharp the distribution is. The `weights` array is normalized to sum to 1, so that the weighted sum of distances is calculated correctly. The final score is calculated as a weighted sum of the distances and the angle difference to the best angle.

To change, adjust and test the scoring function more easily, I created some helper geoms for each angle and distance pair. These geoms' positions are in front of the car which has a distance of `max_scoring_distance` and the angle is calculated based on the current angle and the ray angles. This way, I can visualize the distances to the obstacles and the steering angles in the simulation environment. Each geom has a color in each step:
**Green** represents the selected angle.
**Red** represents the angle that is set by the Pure Pursuit controller.
**Grey** represents the angels whose distances are considered for scoring.
**Black** represents the rest of them.

They can be open and closed by setting the `show_helper_geoms` variable to `True` or `False`.

For final parameters, I increased the `distances_length` to 121 and keep it there. Still, the scoring function can be better, and I can improve it further by adjusting the weight factors or using a different function for scoring. However, I decided to keep it simple for now and focus on the overall performance of the car in the simulation.

# Calculating the Speed

In addition to taking steering angle into account to calculate the speed of the car, I used a simple formula that takes into account the obstacle distance of the steering angle and the difference between the steering angle and the goal steering angle which is calculated using the Pure Pursuit controller. The reason I used these parameters is decreasing the speed when there is an obstacle. The obstacle distance is generally high and the difference between the steering angle and the goal steering angle is generally 0 when there are no obstacles. So, the velocity doesn't decrease when there are no obstacles. The speed is calculated as follows:

```python
def calculate_velocity(angle, target_angle=0, max_velocity=1.0, min_velocity=0.1, distance=3.0):
    # Calculate the velocity according to the closeness to the best angle
    angle_difference = abs(angle - target_angle) / np.pi
    # difference = (1 - angle_difference)

    # The velocity function must be e^-5*angle_difference
    velocity = max_velocity * np.exp(-5*angle_difference)

    # If the distance is too small, decrease the velocity
    velocity = np.log(1 + distance*(np.e-1)/max_scoring_distance) * velocity

    if velocity < min_velocity:
        velocity = min_velocity
    return velocity

```

For similar reasons, I used a logarithmic function to calculate the velocity based on the distance to the closest obstacle. The `np.log(1 + distance*(np.e-1)/max_scoring_distance)` part ensures that the velocity decreases logarithmically with the distance. The `max_velocity` and `min_velocity` parameters define the maximum and minimum velocities of the car, respectively. Again, for the angle difference, I used a simple exponential function to calculate the velocity based on the angle difference to the goal steering angle. The `np.exp(-5*angle_difference)` part ensures that the velocity decreases exponentially with the angle difference. The `max_velocity` and `min_velocity` parameters define the maximum and minimum velocities of the car, respectively.

# Ray Casting from the Car

One problem that I encountered was ray casting from inside of the car. Since the car is a rigid body, the rays are cast from the center of the car, which can lead to incorrect distances to obstacles. To solve this problem, I decided to cast the rays from the front of the car instead of the center. This way, I can ensure that the distances to obstacles are calculated correctly and the car avoids obstacles effectively. I created an `offset` value and added it to the ray_start position to cast the rays from the front of the car.

```python
    yaw = quat_to_yaw(d.xquat[car_buddy.id])
    ray_dir = [np.cos(yaw), np.sin(yaw), 0]
    offset = 0.4
    ray_start = [
        car_pos[0] + ray_dir[0] * offset,
        car_pos[1] + ray_dir[1] * offset,
        car_pos[2]
    ]
```

This way, the rays are cast from the front of the car, which ensures that the distances to obstacles are calculated correctly. The `offset` value is set to 0.4, which is a reasonable value to cast the rays from the front of the car. I also made sure that the rays are cast in the direction of the car's orientation by using the yaw angle of the car's quaternion.

I run a couple of tests and decided that raycasting from the front of the car is not the very best solution, since the car can still hit obstacles that are not in front of it. So, I decided to cast the rays from the center of the car and use the distance to the closest obstacle to calculate the speed. This way, I can ensure that the car avoids obstacles effectively while following the path. To avoid obscure results, I change the class of the each geom which is 0 to 2, and I change the body excluding parameter of the `mujoco.mj_multiRay` function to [1,1,0,1,1,1], which means it will ignore the body with class 2, which is the car body. but since some geoms have class 1 in the car, I change the element with the index of 1 from 1 to 0 too. So, every geom of the car will be ignored in raycasting.

```python
    geomgroup = np.ones(6, dtype=np.uint8)
    geomgroup[1] = 0  # Set group 1 (car) to 0 to exclude
    geomgroup[2] = 0

    mujoco.mj_multiRay(m, d, ray_start, ray_dirs, geomgroup, 1, -1, geomid, dist, nray, max_scoring_distance)
```

This way, I can ensure that the car avoids obstacles effectively while following the path. The `geomgroup` parameter is used to exclude the car body from raycasting, which ensures that the distances to obstacles are calculated correctly. But there is still a problem. The car can still hit obstacles that are not in front of it, (generally, from side) since the rays are cast from the center of the car. Then I thought that I can cast the rays from the back of the car as well, I just needed to set the offset value to an negative value instead of 0. Thus, we can avoid from side obstacles while keeping the casting range as small and as dense as possible. For real case scenarios, for example a lidar sensor that is mounted on the front of the car, can have a system of lenses so that the rays can look like they casted from the back of the car, or the lidar can cast beneath and the back of the car.

```python
    offset = -0.4
    ray_start = [
        car_pos[0] + ray_dir[0] * offset,
        car_pos[1] + ray_dir[1] * offset,
        car_pos[2]
    ]
    .
    .
    .
    .

     mujoco.mj_multiRay(m, d, ray_start, ray_dirs, geomgroup, 1, -1, geomid, dist, nray, max_scoring_distance - offset)

    # replace all -1 distances with max_scoring_distance - offset
    dist[dist < 0] = max_scoring_distance - offset
    # subtract the offset from the distances
    dist += offset
```

I subtracted the `offset` value from the `max_scoring_distance` , then added bact to the distances after raycasting. This way, I can ensure that the distances are calculated correctly and the car avoids obstacles effectively while following the path. The `offset` value is set to -0.4, which is a reasonable value to cast the rays from the back of the car. I also made sure that the rays are cast in the direction of the car's orientation by using the yaw angle of the car's quaternion.

# Back Mode

As much as I wanted to avoid for having states for the drive become much smoother, I needed to implement a back mode for the car. It only activates when the best angle's distance is less than 0.7 and deactivates when the distance is greater than 2.0. When the back mode is activated, the steering and speed values are negated and has a lower limit. It can be improved, but those cases are very rare and this mode handles them well enough.

# Conclusion

To be fair, most of the time on this project was spent on finding the best parameters and the best scoring function and the speed calculation. I tried many different approaches and finally found a combination that works well for my simulation. There are still cases when a path cannot be found, or the car hits the obstacles or the wall, or the car is lost the path. But those cases become edge cases instead of the main cases. In the future, different algorithms and parameters can be tested or current algorithms can be written in a more efficient way such as writing in C++ since it is much faster than Python.


https://github.com/user-attachments/assets/1541cdf8-8772-4321-bfa7-14b882cbbe1a




https://github.com/user-attachments/assets/0fdc5453-ad7e-4640-94de-418f648c0677



