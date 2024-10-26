import mujoco
import glfw
import numpy as np
import random
import matplotlib
import matplotlib.pyplot as plt
import math

class PIDController:
    def __init__(self, kp, ki, kd, setpoint=0):
        self.kp = kp  # Proportional gain
        self.ki = ki  # Integral gain
        self.kd = kd  # Derivative gain
        self.setpoint = setpoint  # Target value
        self.previous_error = 0
        self.integral = 0

    def compute(self, current_value, dt):
        
        # Compute error between current value and setpoint
        error = self.setpoint - current_value
        
        # Proportional term
        p_term = self.kp * error
        
        # Integral term
        self.integral += error * dt
        i_term = self.ki * self.integral
        
        # Derivative term
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
    
    T = [Node(start_pos)]  # Initialize the tree with the start node
    path_found = None
    
    for _ in range(N):
        
        xrand = sample_random_position(goal_area)  # Sample a random position in the environment
        xnear = nearest(T, xrand)  # Find the nearest node in the tree
       
        if is_collision_free(xrand, walls, outside_walls, safety_margin=0.01):
            
            # Set PID controller setpoints for the sampled target
            pid_x.setpoint, pid_y.setpoint = xrand[0], xrand[1]

            # Simulate the new position with iterative control steps
            xe = simulate(pid_x, pid_y, xnear.position, xrand, tolerance=tolerance, max_speed=10, max_steps=500, dt=0.01, logging=logging)

            # Check if the path segment from xnear to xe is collision-free
            if is_collision_free_line(xnear.position, xe, walls, outside_walls, num_samples=100):
                # If the line goes through the goal area, project onto the left wall of the goal area
                if line_goes_through_goal(xnear.position, xe, goal_area):
                    xe = project_to_left_wall(xe, goal_area)

                new_node = Node(xe, parent=xnear)
                new_node.control = (pid_x.compute(xe[0], dt=0.01), pid_y.compute(xe[1], dt=0.01))
                T.append(new_node)

                # Check if xe is inside the goal area
                if is_in_goal_area(xe, goal_area):
                    path_found = construct_path(new_node)
                    if logging:  # Log the success
                        print("Reached goal area")
                    break

    # Visualize the final tree after all nodes are created
    if plot:
        visualize_final_tree(T, path_found, goal_area, walls, outside_walls, start_pos)

    return path_found, T  # Return None if no path is found

def line_goes_through_goal(point1, point2, goal_area, num_samples=10):
    # Sample points along the line from point1 to point2
    for t in range(1, num_samples):
        # Interpolating points between point1 and point2
        x = point1[0] + (point2[0] - point1[0]) * t / num_samples
        y = point1[1] + (point2[1] - point1[1]) * t / num_samples
        sample_point = (x, y)
        
        # Check if the sampled point is in the goal area
        if is_in_goal_area(sample_point, goal_area):
            return True  # The line goes through the goal area
    
    return False  # No points along the line were in the goal area

def project_to_left_wall(point, goal_area):
    # Project a point onto the left wall of the goal area
    x_min = min([coord[0] for coord in goal_area])
    y_min = min([coord[1] for coord in goal_area])
    y_max = max([coord[1] for coord in goal_area])

    # Clamp the y-coordinate to be within the bounds of the goal area vertically
    y_projected = max(min(point[1], y_max), y_min)
    
    # Set the x-coordinate to the left wall (x_min)
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
        
        # Calculate the control signal based on PID output
        control_x = pid_x.compute(current_position[0], dt)
        control_y = pid_y.compute(current_position[1], dt)
        
        # Create control vector and calculate its magnitude
        control_vector = np.array([control_x, control_y])
        control_magnitude = np.linalg.norm(control_vector)
        
        # Calculate distance to target
        distance_to_target = np.linalg.norm(target_position - current_position)
        
        # Determine desired speed based on distance to target
        if distance_to_target < slowdown_distance:
            desired_speed = min_speed + (max_speed - min_speed) * (distance_to_target / slowdown_distance)
        else:
            desired_speed = max_speed

        # Scale the control vector to match the desired speed, if control is non-zero
        if control_magnitude > 0:
            control_vector_normalized = control_vector / control_magnitude
            control_vector = control_vector_normalized * desired_speed

        # Update the position based on the adjusted control action and time step
        new_x = current_position[0] + control_vector[0] * dt
        new_y = current_position[1] + control_vector[1] * dt
        current_position = np.array([new_x, new_y])

        # Debug statements for tracing
        if logging:
            print(f"Step {step}: Position={current_position}, Distance to Target={distance_to_target}, Desired Speed={desired_speed}")

        # Check if within tolerance of the target position
        if distance_to_target <= tolerance:
            return current_position # Return position and path if close enough to the target
    
    return current_position # Return the closest position and path if max_steps is reached

def point_line_distance(point, line_start, line_end):
    
    # Calculate the distance from `point` to the line defined by `line_start` and `line_end`
    line_vec = np.array(line_end) - np.array(line_start)
    point_vec = np.array(point) - np.array(line_start)
    line_len = np.linalg.norm(line_vec)
    line_unitvec = line_vec / line_len
    proj_length = np.dot(point_vec, line_unitvec)
    proj_point = np.array(line_start) + proj_length * line_unitvec
    distance = np.linalg.norm(point - proj_point)
    
    # Check if the projected point lies on the line segment
    if 0 <= proj_length <= line_len:
        return distance
    else:
        # Distance to the nearest endpoint if the projection is outside the line segment
        return min(np.linalg.norm(point - np.array(line_start)), np.linalg.norm(point - np.array(line_end)))

def is_collision_free(xe, walls, outside_walls, safety_margin=0.1):
    
    # Check if the new position is out of bounds
    if xe[0] < -0.5 or xe[0] > 1.5 or xe[1] < -0.4 or xe[1] > 0.4:
        return False  # Outside the map bounds
    
    # Check if the new position collides with any obstacles (walls)
    for wall, coordinates in walls.items():
        # Coordinates is a list of corner points of the wall (assumed rectangular here)
        x_min = min([coord[0] for coord in coordinates]) - safety_margin
        x_max = max([coord[0] for coord in coordinates]) + safety_margin
        y_min = min([coord[1] for coord in coordinates]) - safety_margin
        y_max = max([coord[1] for coord in coordinates]) + safety_margin

        # Check if the new position xe lies within the bounds of the wall with the safety margin
        if x_min <= xe[0] <= x_max and y_min <= xe[1] <= y_max:
            return False  # Collision with the wall (including safety margin)
    
    # Check for collisions with the outside walls
    for wall_line in outside_walls:
        line_start, line_end = wall_line
        if point_line_distance(xe, line_start, line_end) <= safety_margin:
            return False  # Collision with outside wall

    return True

def is_collision_free_line(p1, p2, walls, outside_walls, num_samples=100):
    
    for t in np.linspace(0, 1, num_samples):
        # Interpolate between p1 and p2
        point = (1 - t) * np.array(p1) + t * np.array(p2)
        
        if not is_collision_free(point, walls, outside_walls):
            return False  # If any point on the line collides, return False

    return True  # If no collisions, return True

def construct_path(node):
    # Reconstruct the path from the goal node to the start node
    path = []
    while node is not None:
        path.append(node.position)
        node = node.parent
    return path[::-1]  # Return the path from start to goal

def smooth_path(path, walls, outside_walls, max_attempts=20):
    
    smoothed_path = list(path)  # Create a copy of the path

    for _ in range(max_attempts):  # Perform smoothing for a fixed number of attempts
        if len(smoothed_path) <= 2:
            break  # If the path has only start and end, it's already smooth

        # Randomly select two points in the path
        i, j = sorted(random.sample(range(len(smoothed_path)), 2))

        # Check if a direct path between these two points is collision-free
        if is_collision_free_line(smoothed_path[i], smoothed_path[j], walls, outside_walls):
            # Remove the intermediate points and connect i to j directly
            smoothed_path = smoothed_path[:i + 1] + smoothed_path[j:]
    
    return smoothed_path

def plot_path_with_boundaries_and_mixed_obstacles(paths, walls=None, goal_area=None, outside_walls=None):
    
    # Set up the plot
    plt.figure(figsize=(8, 6))
    
    # Plot 2D walls as boxes if provided
    if walls:
        for wall, coordinates in walls.items():
            wall_polygon = plt.Polygon(coordinates, color='red', alpha=0.5)
            plt.gca().add_patch(wall_polygon)

    # Plot outside walls as lines if provided
    if outside_walls:
        for wall in outside_walls:
            plt.plot([wall[0][0], wall[1][0]], [wall[0][1], wall[1][1]], 'k-', lw=2)

    # Plot the goal area as a 2D box if provided
    if goal_area:
        goal_polygon = plt.Polygon(goal_area, color='green', alpha=0.3)
        plt.gca().add_patch(goal_polygon)

    # Plot each path in the list with a random color
    for path in paths:
        path = np.array(path)  # Ensure path is a numpy array
        random_color = [random.random() for _ in range(3)]  # Generate a random color
        plt.plot(path[:, 0], path[:, 1], marker='o', color=random_color)

        # Plot the start and goal positions for each path
        plt.plot(path[0, 0], path[0, 1], 'go', markersize=10)  # Start point
        plt.plot(path[-1, 0], path[-1, 1], 'ro', markersize=10)  # Goal point

    # Set plot limits
    plt.xlim(-0.6, 1.6)
    plt.ylim(-0.5, 0.5)

    # Add labels and title
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.title("Kinodynamic RRT Paths with Map Boundaries and Obstacles")

    # Show the plot
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
    
    ball_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ball")
    dt = 0.01
    time_data, deviation_data = [], []
    time_elapsed = 0
    max_speed, min_speed = 15, 1.25
    slowdown_distance = .75
    
    # Initialize plot if plotting is enabled
    if plot_enabled:
        fig, ax, line_dev = initialize_plot()

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
            
            if distance_to_goal < 0.075:
                if logging:
                    print("Reached node", i + 1)
                break

            desired_speed = update_speed(distance_to_goal, slowdown_distance, max_speed, min_speed)

            if logging:
                print(f"Distance to goal: {distance_to_goal}, Desired speed: {desired_speed}")
            
            control_x, control_y = apply_pid_control(pid_x, pid_y, ball_pos, dt, desired_speed)

            if logging:
                print(f"Control signals: control_x={control_x}, control_y={control_y}")

            data.ctrl[0], data.ctrl[1] = control_x, control_y
            mujoco.mj_step(model, data)
            
            # Render the scene if rendering is enabled
            if render_enabled:
                render_scene(model, data, options, scene, context, viewport, camera, window)

            # Plot deviation if plotting is enabled
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
    
    # Close the plot if it was enabled
    if plot_enabled:
        plt.close() 

    return time_elapsed

def visualize_final_tree(tree, path, goal_area=None, walls=None, outside_walls=None, start_pos=None):
    
    plt.figure(figsize=(8, 6))
    
    # Plot 2D walls as polygons if provided
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

    # Plot outside walls as lines if provided
    if outside_walls:
        for wall in outside_walls:
            plt.plot([wall[0][0], wall[1][0]], [wall[0][1], wall[1][1]], 'k-', lw=2)

    # Plot the goal area as a 2D box if provided
    if goal_area:
        goal_polygon = plt.Polygon(goal_area, color='green', alpha=0.3)
        plt.gca().add_patch(goal_polygon)

    # Plot the tree nodes and edges
    for node in tree:
        if node.parent:
            # Draw the edge from parent to node
            plt.plot([node.position[0], node.parent.position[0]],
                     [node.position[1], node.parent.position[1]], 'b-', alpha=0.5)  # Blue edges for tree
        # Draw the node itself
        plt.plot(node.position[0], node.position[1], 'bo', markersize=2)  # Blue nodes

    # Plot the start position if provided
    if start_pos:
        plt.plot(start_pos[0], start_pos[1], 'go', label='Start', markersize=10)  # Green dot for start position

    # Plot the path if one was found
    if path:
        path = np.array(path)  # Ensure path is a numpy array
        plt.plot(path[:, 0], path[:, 1], 'c-', linewidth=2, label='Path')  # Cyan line for the path
        plt.plot(path[-1, 0], path[-1, 1], 'ro', label='Goal', markersize=10)  # Red dot for goal position

    # Configure the plot
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

    # Set viewport
    viewport = mujoco.MjrRect(0, 0, 1200, 900)

    # Return the window, camera, scene, context, options, and viewport for rendering
    return window, camera, scene, context, options, viewport

def model_creation(start_pos, goal_area, walls, outside_walls):
   
    # Load the MuJoCo model
    model = mujoco.MjModel.from_xml_path("ball_square.xml")
    data = mujoco.MjData(model)
    
    # Perform Kinodynamic-RRT to find a path that reaches the goal area
    pid_x = PIDController(kp=.5, ki=0.0, kd=0.5)
    pid_y = PIDController(kp=.5, ki=0.0, kd=0.5)
    path, tree = kinodynamic_rrt(pid_x, pid_y, start_pos, goal_area, walls, outside_walls=outside_walls)
    
    if path:
        path = smooth_path(path, walls, outside_walls)  # Smooth the path
        plot_path_with_boundaries_and_mixed_obstacles([path], walls, goal_area, outside_walls)
        
        pid_x = PIDController(kp=.58, ki=0.0, kd=0.5)
        pid_y = PIDController(kp=.58, ki=0.0, kd=0.5)
        
        # Initialize the window and visualization structures
        window, camera, scene, context, options, viewport = init_glfw_window(model)
        
        # Control the ball to follow the path
        move_ball_along_path_with_pid(pid_x, pid_y, model, data, path, window, scene, context, options, viewport, camera)
    
    else:
        print("No path found")

    plt.close('all')
    glfw.terminate()
    
    return

def tree_visualization(start_pos, walls, goal_area, outside_walls):
    
    pid_x = PIDController(kp=.45, ki=0.0, kd=0.5)
    pid_y = PIDController(kp=.45, ki=0.0, kd=0.5)
    
    for trial in range(5):
        seed = generate_random_seed()
        random.seed(seed)
        np.random.seed(seed)
        path, tree = kinodynamic_rrt(pid_x, pid_y, start_pos, goal_area, walls, outside_walls=outside_walls, tolerance=0.15, N=1000, plot=True, logging=False)
        print(f"Trial {trial + 1} - Path: {'Found' if path else 'Not Found'}")

    return

def generate_random_seed():
    seed = random.randint(0, 2**32 - 1)  # Generate a random integer seed within a 32-bit range
    return seed

def run_trials(start_pos, goal_area, walls, outside_walls, num_trials, Tmax):
    
    success_count = 0
    plan_time = -1
    paths = []
    
    for trial in range(num_trials):
        
        print(f"Running trial {trial + 1}/{num_trials}...")
        seed = generate_random_seed()
        
        # Load the MuJoCo model
        model = mujoco.MjModel.from_xml_path("ball_square.xml")
        data = mujoco.MjData(model)
        
        pid_x = PIDController(kp=.58, ki=0.0, kd=0.5)
        pid_y = PIDController(kp=.58, ki=0.0, kd=0.5)

        # Generate a path using kinodynamic RRT
        path, tree = kinodynamic_rrt(pid_x, pid_y, start_pos, goal_area, walls, outside_walls=outside_walls, tolerance=0.15, N=1000, plot=False, logging=False)
        
        if path:
            
            # Smooth the path to avoid obstacles
            path = smooth_path(path, walls, outside_walls)
            paths.append(path)
            
            # Calculate planning time for the smoothed path
            plan_time = move_ball_along_path_with_pid(pid_x, pid_y, model, data, path, plot_enabled=False, render_enabled=False, logging=False, Tmax=Tmax)
            print(f"Planning time: {plan_time:.2f} seconds")

            # Check if the planning time is within the allowed limit
            if plan_time <= Tmax:
                success_count += 1
                print("Successfully reached the goal area!")
            else:
                print("Failed to reach the goal area within the time limit.")
        else:
            print("No path found.")

    # Report the success rate after all trials
    success_rate = (success_count / num_trials) * 100
    print(f"Success rate over {num_trials} trials: {success_rate:.2f}%")
    plot_path_with_boundaries_and_mixed_obstacles(paths, walls, goal_area, outside_walls)
    
    return

def main():
    # Define the goal area (a rectangular region)
    goal_area = [[0.9, -0.3], [0.9, 0.3], [1.1, 0.3], [1.1, -0.3]]
    
    # Define outside walls as lines
    outside_walls = [
        [[-0.5, -0.4], [-0.5, 0.4]],
        [[1.5, -0.4], [1.5, 0.4]],
        [[-0.5, 0.4], [1.5, 0.4]],
        [[-0.5, -0.4], [1.5, -0.4]]
    ]

    # Define the middle obstacle
    walls = {
        "wall_3": [[0.5, -0.15], [0.5, 0.15], [0.6, 0.15], [0.6, -0.15]]
    }

    # Define the start position
    start_pos = [0, 0]  # Starting at the origin

    while True:
        try:
            print("\nMenu:")
            print("1. Model Creation")
            print("2. Tree Visualization")
            print("3. Planning Time")
            print("4. Quit")
            
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
                for Tmax in [30, 20, 10, 5]:
                    print(f'Starting {num_trials} trails for {Tmax} seconds')
                    run_trials(start_pos, goal_area, walls, outside_walls, num_trials, Tmax)
            elif choice == 4:
                print("Exiting the program. Goodbye!")
                break
            else:
                print("Invalid option. Please try again.")
        
        except KeyboardInterrupt:
            print("\nProgram interrupted. Exiting gracefully.")
            break
    return

if __name__ == "__main__":
    main()
