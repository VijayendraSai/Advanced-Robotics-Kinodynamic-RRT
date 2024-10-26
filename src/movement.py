import mujoco
import glfw
import numpy as np
import random
import matplotlib
import matplotlib.pyplot as plt
import math

class PIDController:
    def __init__(self, kp, ki, kd, setpoint=0):
        self.kp = kp  
        self.ki = ki  
        self.kd = kd  
        self.setpoint = setpoint
        self.previous_error = 0
        self.integral = 0

    def compute(self, current_value, dt):
        
        # error between points
        error = self.setpoint - current_value
        
        # kp
        p_term = self.kp * error
        
        # ki
        self.integral += error * dt
        i_term = self.ki * self.integral
        
        # kd
        d_term = self.kd * (error - self.previous_error) / dt
        self.previous_error = error
        
        # Return the control output
        return p_term + i_term + d_term

class Node:
    def __init__(self, position, parent=None):
        self.position = position
        self.parent = parent
        self.control = None

def is_in_goal_area(point, goal_area):
    
    x_min = min([coord[0] for coord in goal_area])
    x_max = max([coord[0] for coord in goal_area])
    y_min = min([coord[1] for coord in goal_area])
    y_max = max([coord[1] for coord in goal_area])

    return x_min <= point[0] <= x_max and y_min <= point[1] <= y_max

def kinodynamic_rrt(pid_x, pid_y, start_pos, goal_area, walls, outside_walls=None, tolerance=0.10, N=1000, plot=True, logging=True):
    
    T = [Node(start_pos)]    
    path_found = None
   
    # run the search for points
    for _ in range(N):
        
        xrand = sample_random_position(goal_area)
        xnear = nearest(T, xrand) 
        
        # check that the point is collision free with the safety marging
        if is_collision_free(xrand, walls, outside_walls, safety_margin=0.01):
            
            # simulate getting to the random point
            pid_x.setpoint, pid_y.setpoint = xrand[0], xrand[1]
            xe = simulate(pid_x, pid_y, xnear.position, xrand, tolerance=tolerance, max_speed=10, max_steps=500, dt=0.01, logging=logging)

            # check the line between the points is collison free
            if is_collision_free_line(xnear.position, xe, walls, outside_walls, num_samples=100):
                
                # project back to the goal area
                if line_goes_through_goal(xnear.position, xe, goal_area):
                    xe = project_to_left_wall(xe, goal_area)

                new_node = Node(xe, parent=xnear)
                new_node.control = (pid_x.compute(xe[0], dt=0.01), pid_y.compute(xe[1], dt=0.01))
                T.append(new_node)

                # stop condition
                if is_in_goal_area(xe, goal_area):
                    path_found = construct_path(new_node)
                    if logging:
                        print("Reached goal area")
                    break

    # see the full search tree
    if plot:
        visualize_final_tree(T, path_found, goal_area, walls, outside_walls, start_pos)

    return path_found, T  # Return None if no path is found

def line_goes_through_goal(point1, point2, goal_area, num_samples=10):
    
    # sample points between line
    for t in range(1, num_samples):
        
        # interpolating points for testing
        x = point1[0] + (point2[0] - point1[0]) * t / num_samples
        y = point1[1] + (point2[1] - point1[1]) * t / num_samples
        sample_point = (x, y)
        
        # Check if the sampled point is in the goal area
        if is_in_goal_area(sample_point, goal_area):
            return True 

    return False

def project_to_left_wall(point, goal_area):
    
    # project the point onto the left wall of goal area
    x_min = min([coord[0] for coord in goal_area])
    y_min = min([coord[1] for coord in goal_area])
    y_max = max([coord[1] for coord in goal_area])
    y_projected = max(min(point[1], y_max), y_min)
    
    return (x_min, y_projected)

def sample_random_position(goal_area=None, goal_bias=1, bias_strength=0.5):
    
    x = random.uniform(-0.5, 1.5)
    y = random.uniform(-0.4, 0.4)
    
    return np.array([x, y])

def nearest(T, xrand):
    return min(T, key=lambda node: np.linalg.norm(node.position - xrand))

def simulate(pid_x, pid_y, position, target_position, tolerance=0.05, max_steps=100, dt=0.01, max_speed=3, min_speed=1, slowdown_distance=1.0, logging=True):
    
    current_position = np.array(position)

    for step in range(max_steps):
        
        # calculate the control
        control_x = pid_x.compute(current_position[0], dt)
        control_y = pid_y.compute(current_position[1], dt)
        
        # create control vector and calculate its magnitude
        control_vector = np.array([control_x, control_y])
        control_magnitude = np.linalg.norm(control_vector)
        distance_to_target = np.linalg.norm(target_position - current_position)
    
        # determine desired speed based on distance to target
        if distance_to_target < slowdown_distance:
            desired_speed = min_speed + (max_speed - min_speed) * (distance_to_target / slowdown_distance)
        else:
            desired_speed = max_speed

        # scale the controls based on speed
        if control_magnitude > 0:
            control_vector_normalized = control_vector / control_magnitude
            control_vector = control_vector_normalized * desired_speed

        # update the position based on the adjusted control action and time step
        new_x = current_position[0] + control_vector[0] * dt
        new_y = current_position[1] + control_vector[1] * dt
        current_position = np.array([new_x, new_y])

        if logging:
            print(f"Step {step}: Position={current_position}, Distance to Target={distance_to_target}, Desired Speed={desired_speed}")

        if distance_to_target <= tolerance:
            return current_position
    
    return current_position

def point_line_distance(point, line_start, line_end):
    
    line_vec = np.array(line_end) - np.array(line_start)
    point_vec = np.array(point) - np.array(line_start)
    line_len = np.linalg.norm(line_vec)
    line_unitvec = line_vec / line_len
    
    proj_length = np.dot(point_vec, line_unitvec)
    proj_point = np.array(line_start) + proj_length * line_unitvec
    distance = np.linalg.norm(point - proj_point)
    
    if 0 <= proj_length <= line_len:
        return distance
    else:
        return min(np.linalg.norm(point - np.array(line_start)), np.linalg.norm(point - np.array(line_end)))

def is_collision_free(xe, walls, outside_walls, safety_margin=0.1):
    
    if xe[0] < -0.5 or xe[0] > 1.5 or xe[1] < -0.4 or xe[1] > 0.4:
        return False
    
    for wall, coordinates in walls.items():
        x_min = min([coord[0] for coord in coordinates]) - safety_margin
        x_max = max([coord[0] for coord in coordinates]) + safety_margin
        y_min = min([coord[1] for coord in coordinates]) - safety_margin
        y_max = max([coord[1] for coord in coordinates]) + safety_margin

        if x_min <= xe[0] <= x_max and y_min <= xe[1] <= y_max:
            return False
    
    for wall_line in outside_walls:
        line_start, line_end = wall_line
        if point_line_distance(xe, line_start, line_end) <= safety_margin:
            return False

    return True

def is_collision_free_line(p1, p2, walls, outside_walls, num_samples=100):
    
    for t in np.linspace(0, 1, num_samples):
        
        # interpolate between p1 and p2
        point = (1 - t) * np.array(p1) + t * np.array(p2)
        if not is_collision_free(point, walls, outside_walls):
            return False

    return True

def construct_path(node):
    
    # reconstruct the path from the goal node to the start node
    path = []
    while node is not None:
        path.append(node.position)
        node = node.parent
    return path[::-1]

def smooth_path(path, walls, outside_walls, max_attempts=20):
    
    smoothed_path = list(path)

    for _ in range(max_attempts):
        if len(smoothed_path) <= 2:
            break 

        # randomly select two points in the path
        i, j = sorted(random.sample(range(len(smoothed_path)), 2))

        # update the path when collison free
        if is_collision_free_line(smoothed_path[i], smoothed_path[j], walls, outside_walls):
            smoothed_path = smoothed_path[:i + 1] + smoothed_path[j:]
    
    return smoothed_path

def plot_path_with_boundaries_and_mixed_obstacles(paths, walls=None, goal_area=None, outside_walls=None):
    
    plt.figure(figsize=(8, 6))
    
    # plot 2d walls as boxes if provided
    if walls:
        for wall, coordinates in walls.items():
            wall_polygon = plt.Polygon(coordinates, color='red', alpha=0.5)
            plt.gca().add_patch(wall_polygon)

    # plot outside walls as lines
    if outside_walls:
        for wall in outside_walls:
            plt.plot([wall[0][0], wall[1][0]], [wall[0][1], wall[1][1]], 'k-', lw=2)

    # plot the goal area as a 2d box
    if goal_area:
        goal_polygon = plt.Polygon(goal_area, color='green', alpha=0.3)
        plt.gca().add_patch(goal_polygon)

    # plot each path
    for path in paths:
        path = np.array(path)
        random_color = [random.random() for _ in range(3)]
        plt.plot(path[:, 0], path[:, 1], marker='o', color=random_color)

        # Plot the start and goal positions for each path
        plt.plot(path[0, 0], path[0, 1], 'go', markersize=10)
        plt.plot(path[-1, 0], path[-1, 1], 'ro', markersize=10)

    plt.xlim(-0.6, 1.6)
    plt.ylim(-0.5, 0.5)
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.title("Kinodynamic RRT Paths with Map Boundaries and Obstacles")
    plt.grid(True)
    plt.show()
    
    return

def initialize_plot():
    
    plt.ion()
    fig, ax = plt.subplots()
    line_dev, = ax.plot([], [], 'b-', label='Deviation')
    ax.set_xlim(0, 10)
    ax.set_ylim(-0.5, 0.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Deviation from Line (m)')
    ax.legend()
    plt.grid(True)
    
    return fig, ax, line_dev

def calculate_deviation(A, B, P):
    
    deviation = (B[0] - A[0]) * (A[1] - P[1]) - (A[0] - P[0]) * (B[1] - A[1]) / np.sqrt((B[0] - A[0]) ** 2 + (B[1] - A[1]) ** 2)
    
    return deviation

def update_speed(distance_to_goal, slowdown_distance, max_speed, min_speed):
    
    if distance_to_goal < slowdown_distance:
        return max(min_speed, (distance_to_goal / slowdown_distance) * max_speed)
    
    return max_speed

def apply_pid_control(pid_x, pid_y, ball_pos, dt, desired_speed):
    
    control_x = pid_x.compute(ball_pos[0], dt)
    control_y = pid_y.compute(ball_pos[1], dt)
    control_vector = np.array([control_x, control_y])
    control_magnitude = np.linalg.norm(control_vector)
    
    if control_magnitude > 0:
        control_vector_normalized = control_vector / control_magnitude
        control_x = control_vector_normalized[0] * desired_speed
        control_y = control_vector_normalized[1] * desired_speed
    
    return control_x, control_y

def render_scene(model, data, options, scene, context, viewport, camera, window):
    
    mujoco.mjv_updateScene(model, data, options, None, camera, mujoco.mjtCatBit.mjCAT_ALL, scene)
    mujoco.mjr_render(viewport, scene, context)
    glfw.poll_events()
    glfw.swap_buffers(window)
    
    return

def move_ball_along_path_with_pid(pid_x, pid_y, model, data, path, window=None, scene=None, context=None, options=None, viewport=None, camera=None, plot_enabled=True, render_enabled=True, logging=True, Tmax=120):
   
    # set the parameters for the model
    ball_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ball")
    dt = 0.01
    time_data, deviation_data = [], []
    time_elapsed = 0
    max_speed, min_speed = 15, 1.25
    slowdown_distance = .75
    
    # create the live plot for deviation
    if plot_enabled:
        fig, ax, line_dev = initialize_plot()
    
    # loop through each point and set the controllers and move until the next point is reached
    for i in range(len(path) - 1):
        
        start_pos = np.array(path[i])
        end_pos = np.array(path[i + 1])
        line_direction = end_pos - start_pos
        line_length = np.linalg.norm(line_direction)
        line_direction_normalized = line_direction / line_length
        pid_x.setpoint, pid_y.setpoint = end_pos[0], end_pos[1]

        while True:
            
            ball_pos = data.xpos[ball_id][:2]
            distance_to_goal = np.linalg.norm(end_pos - ball_pos)
            
            # break when a point is reached
            if distance_to_goal < 0.075:
                if logging:
                    print("Reached node", i + 1)
                break
            
            # update the speed based on the slowdown distance
            desired_speed = update_speed(distance_to_goal, slowdown_distance, max_speed, min_speed)

            if logging:
                print(f"Distance to goal: {distance_to_goal}, Desired speed: {desired_speed}")
           
            # update the controllers based on new ball position
            control_x, control_y = apply_pid_control(pid_x, pid_y, ball_pos, dt, desired_speed)

            if logging:
                print(f"Control signals: control_x={control_x}, control_y={control_y}")

            data.ctrl[0], data.ctrl[1] = control_x, control_y
            mujoco.mj_step(model, data)
            
            # render the 3d
            if render_enabled:
                render_scene(model, data, options, scene, context, viewport, camera, window)

            # plot deviation from the line
            if plot_enabled:
                deviation = calculate_deviation(start_pos, end_pos, ball_pos)
                deviation_data.append(deviation)
                time_data.append(time_elapsed)
                line_dev.set_xdata(time_data)
                line_dev.set_ydata(deviation_data)
                ax.set_xlim(0, max(10, time_elapsed + 1))
                fig.canvas.draw()
                fig.canvas.flush_events()

            time_elapsed += dt

            if time_elapsed > Tmax:
                break
    
    if plot_enabled:
        plt.close(fig) 
        plt.close('all')

    return time_elapsed

def visualize_final_tree(tree, path, goal_area=None, walls=None, outside_walls=None, start_pos=None):
   
    plt.figure(figsize=(8, 6))
    
    if walls:
        if isinstance(walls, dict):
            wall_list = walls.values()
        elif isinstance(walls, list):
            wall_list = walls
        else:
            raise ValueError("walls must be either a dictionary or a list")
        
        for coordinates in wall_list:
            wall_polygon = plt.Polygon(coordinates, color='red', alpha=0.5)
            plt.gca().add_patch(wall_polygon)

    # plot outside walls
    if outside_walls:
        for wall in outside_walls:
            plt.plot([wall[0][0], wall[1][0]], [wall[0][1], wall[1][1]], 'k-', lw=2)

    # plot the goal area
    if goal_area:
        goal_polygon = plt.Polygon(goal_area, color='green', alpha=0.3)
        plt.gca().add_patch(goal_polygon)

    # plot the tree nodes and edges
    for node in tree:
        if node.parent:
            plt.plot([node.position[0], node.parent.position[0]],
                     [node.position[1], node.parent.position[1]], 'b-', alpha=0.5)
        plt.plot(node.position[0], node.position[1], 'bo', markersize=2)

    # plot the start position
    if start_pos:
        plt.plot(start_pos[0], start_pos[1], 'go', label='Start', markersize=10)

    # plot the path
    if path:
        path = np.array(path)
        plt.plot(path[:, 0], path[:, 1], 'c-', linewidth=2, label='Path')
        plt.plot(path[-1, 0], path[-1, 1], 'ro', label='Goal', markersize=10)

    # configure the plot
    plt.xlim(-0.6, 1.6)
    plt.ylim(-0.5, 0.5)
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.title("Kinodynamic RRT Tree with Map Boundaries and Obstacles")
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.show()
    
    return

def init_glfw_window(model):
    
    if not glfw.init():
        raise Exception("Could not initialize glfw")

    window = glfw.create_window(1200, 900, "MuJoCo Simulation", None, None)
    if not window:
        glfw.terminate()
        raise Exception("Could not create glfw window")

    glfw.make_context_current(window)
    
    camera = mujoco.MjvCamera()
    scene = mujoco.MjvScene(model, maxgeom=10000)
    context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)
    options = mujoco.MjvOption()
    
    # camera settings
    camera.distance = 3.0
    camera.elevation = -45.0
    camera.azimuth = 0.0

    viewport = mujoco.MjrRect(0, 0, 1200, 900)
    return window, camera, scene, context, options, viewport

def model_creation(start_pos, goal_area, walls, outside_walls):
   
    # load the MuJoCo model
    model = mujoco.MjModel.from_xml_path("ball_square.xml")
    data = mujoco.MjData(model)
    
    # generate a path using the controllers and kinodyanmic_rrt
    pid_x = PIDController(kp=.5, ki=0.0, kd=0.5)
    pid_y = PIDController(kp=.5, ki=0.0, kd=0.5)
    path, tree = kinodynamic_rrt(pid_x, pid_y, start_pos, goal_area, walls, outside_walls=outside_walls)
    
    if path:
        path = smooth_path(path, walls, outside_walls)
        plot_path_with_boundaries_and_mixed_obstacles([path], walls, goal_area, outside_walls)
        
        pid_x = PIDController(kp=.58, ki=0.0, kd=0.5)
        pid_y = PIDController(kp=.58, ki=0.0, kd=0.5)
        
        # create the window
        window, camera, scene, context, options, viewport = init_glfw_window(model)
        
        # run the 3d simulation with the path and window objects
        move_ball_along_path_with_pid(pid_x, pid_y, model, data, path, window, scene, context, options, viewport, camera)
    
    else:
        print("No path found")

    glfw.terminate()
    plt.close('all')
    
    return

def tree_visualization(start_pos, walls, goal_area, outside_walls):
   
    # set the controllers for the kinodynamic rrt
    pid_x = PIDController(kp=.45, ki=0.0, kd=0.5)
    pid_y = PIDController(kp=.45, ki=0.0, kd=0.5)
    
    for trial in range(5):
        seed = generate_random_seed()
        random.seed(seed)
        np.random.seed(seed)
        
        # run the trail
        path, tree = kinodynamic_rrt(pid_x, pid_y, start_pos, goal_area, walls, outside_walls=outside_walls, tolerance=0.15, N=1000, plot=True, logging=False)
        print(f"Trial {trial + 1} - Path: {'Found' if path else 'Not Found'}")

    return

def generate_random_seed():
    seed = random.randint(0, 2**32 - 1)
    return seed

def run_trials(start_pos, goal_area, walls, outside_walls, num_trials, Tmax):
    
    success_count = 0
    plan_time = -1
    paths = []
    
    for trial in range(num_trials):
        
        print(f"Running trial {trial + 1}/{num_trials}...")
        seed = generate_random_seed()
        
        # load the MuJoCo model
        model = mujoco.MjModel.from_xml_path("ball_square.xml")
        data = mujoco.MjData(model)
        
        pid_x = PIDController(kp=.58, ki=0.0, kd=0.5)
        pid_y = PIDController(kp=.58, ki=0.0, kd=0.5)

        # generate a path using kinodynamic RRT
        path, tree = kinodynamic_rrt(pid_x, pid_y, start_pos, goal_area, walls, outside_walls=outside_walls, tolerance=0.15, N=1000, plot=False, logging=False)
        
        if path:
            
            # smooth the path for less nodes
            path = smooth_path(path, walls, outside_walls)
            paths.append(path)
            
            # record the execution time for the smoothed path
            plan_time = move_ball_along_path_with_pid(pid_x, pid_y, model, data, path, plot_enabled=False, render_enabled=False, logging=False, Tmax=Tmax)
            print(f"Planning time: {plan_time:.2f} seconds")

            # check the time
            if plan_time <= Tmax:
                success_count += 1
                print("Successfully reached the goal area!")
            else:
                print("Failed to reach the goal area within the time limit.")
        else:
            print("No path found.")

    # report the success rate
    success_rate = (success_count / num_trials) * 100
    print(f"Success rate over {num_trials} trials: {success_rate:.2f}%")
    plot_path_with_boundaries_and_mixed_obstacles(paths, walls, goal_area, outside_walls)
    
    return

def main():
    
    # define the goal area, walls, and the starting position
    goal_area = [[0.9, -0.3], [0.9, 0.3], [1.1, 0.3], [1.1, -0.3]]
    outside_walls = [
        [[-0.5, -0.4], [-0.5, 0.4]],
        [[1.5, -0.4], [1.5, 0.4]],
        [[-0.5, 0.4], [1.5, 0.4]],
        [[-0.5, -0.4], [1.5, -0.4]]]
    walls = {"wall_3": [[0.5, -0.15], [0.5, 0.15], [0.6, 0.15], [0.6, -0.15]]}
    start_pos = [0, 0]

    print("\nMenu:")
    print("1. Model Creation")
    print("2. Tree Visualization")
    print("3. Planning Time")
    print("4. Quit")
    
    # allow the user to determine which objective to run
    choice = input("Enter your choice: ")
    if choice.isdigit():
        choice = int(choice)
    else:
        print("Invalid input. Please enter a number.")
        continue

    if choice == 1:
        model_creation(start_pos, goal_area, walls, outside_walls)
    elif choice == 2:
        tree_visualization(start_pos, walls, goal_area, outside_walls)
    elif choice == 3:
        num_trials = 30
        # run all the trails in a loop with changing time
        for Tmax in [30, 20, 10, 5]:
            print(f'Starting {num_trials} trails for {Tmax} seconds')
            run_trials(start_pos, goal_area, walls, outside_walls, num_trials, Tmax)
        exit()
    elif choice == 4:
        print("Exiting the program. Goodbye!")
        break
    else:
        print("Invalid option. Please try again.")

    return

if __name__ == "__main__":
    main()
