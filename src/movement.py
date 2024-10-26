import mujoco
import glfw
import numpy as np
import random
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

def kinodynamic_rrt(pid_x, pid_y, start_pos, goal_area, walls, outside_walls=None, tolerance=0.15, N=1000, plot=True):
    
    T = [Node(start_pos)]  # Initialize the tree with the start node
    path_found = None
    
    for _ in range(N):
        
        xrand = sample_random_position(goal_area)  # Sample a random position in the environment
        xnear = nearest(T, xrand) # Find the nearest node in the tree
       
        if is_collision_free(xrand, walls):

            # Set PID controller setpoints for the sampled target
            pid_x.setpoint, pid_y.setpoint = xrand[0], xrand[1]

            # Simulate the new position with iterative control steps
            xe = simulate(pid_x, pid_y, xnear.position, xrand, tolerance=tolerance, max_steps=50, dt=0.01)

            # Check if the path segment from xnear to xe is collision-free
            if is_collision_free_line(xnear.position, xe, walls, num_samples=100):
                new_node = Node(xe, parent=xnear)
                new_node.control = (pid_x.compute(xe[0], dt=0.01), pid_y.compute(xe[1], dt=0.01))
                T.append(new_node)

                # Check if xe is inside the goal area
                if is_in_goal_area(xe, goal_area):
                    path_found = construct_path(new_node)  # Construct and return the path
                    print("Reached goal area")
                    break

    # Visualize the final tree after all nodes are created
    visualize_final_tree(T, path_found, goal_area, walls, outside_walls, start_pos)

    return path_found, T  # Return None if no path is found

def sample_random_position(goal_area=None, goal_bias=0.2, bias_strength=0.5):
    
    if goal_area and random.random() < goal_bias:
        # Calculate goal area center
        x_center = sum(point[0] for point in goal_area) / len(goal_area)
        y_center = sum(point[1] for point in goal_area) / len(goal_area)
        
        # Sample a random position
        x = random.uniform(-0.5, 1.5)
        y = random.uniform(-0.4, 0.4)
        
        # Bias toward goal center
        x = (1 - bias_strength) * x + bias_strength * x_center
        y = (1 - bias_strength) * y + bias_strength * y_center

        return np.array([x, y])

    x = random.uniform(-0.5, 1.5)
    y = random.uniform(-0.4, 0.4)
    
    return np.array([x, y])

def nearest(T, xrand):
    return min(T, key=lambda node: np.linalg.norm(node.position - xrand))

def choose_control(pid_x, pid_y, xnear, xrand, dt=0.05):
    
    # Desired direction towards xrand
    desired_direction = np.array(xrand) - np.array(xnear)
    desired_distance = np.linalg.norm(desired_direction)
    
    # Normalize direction for unit vector and avoid division by zero
    if desired_distance > 0:
        desired_direction /= desired_distance
    
    # Error in x and y for PID controllers
    error_x = desired_direction[0] * desired_distance
    error_y = desired_direction[1] * desired_distance
    
    # PID outputs for control in x and y
    control_x = pid_x.compute(error_x, dt)
    control_y = pid_y.compute(error_y, dt)
    
    # Return control as a combined vector
    return np.array([control_x, control_y])

def simulate(pid_x, pid_y, position, target_position, tolerance=0.05, max_steps=100, dt=0.01, max_speed=10, min_speed=1, slowdown_distance=1.0):
    
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
        print(f"Step {step}: Position={current_position}, Distance to Target={distance_to_target}, Desired Speed={desired_speed}")

        # Check if within tolerance of the target position
        if distance_to_target <= tolerance:
            return current_position # Return position and path if close enough to the target
    
    return current_position # Return the closest position and path if max_steps is reached

def is_collision_free(xe, walls, safety_margin=.1):
    
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
    
    return True

def is_collision_free_line(p1, p2, walls, num_samples=100):
    
    for t in np.linspace(0, 1, num_samples):
        # Interpolate between p1 and p2
        point = (1 - t) * np.array(p1) + t * np.array(p2)
        
        if not is_collision_free(point, walls):
            return False  # If any point on the line collides, return False

    return True  # If no collisions, return True

def construct_path(node):
    # Reconstruct the path from the goal node to the start node
    path = []
    while node is not None:
        path.append(node.position)
        node = node.parent
    return path[::-1]  # Return the path from start to goal

def smooth_path(path, walls, max_attempts=50):
    
    smoothed_path = list(path)  # Create a copy of the path

    for _ in range(max_attempts):  # Perform smoothing for a fixed number of attempts
        if len(smoothed_path) <= 2:
            break  # If the path has only start and end, it's already smooth

        # Randomly select two points in the path
        i, j = sorted(random.sample(range(len(smoothed_path)), 2))

        # Check if a direct path between these two points is collision-free
        if is_collision_free_line(smoothed_path[i], smoothed_path[j], walls):
            # Remove the intermediate points and connect i to j directly
            smoothed_path = smoothed_path[:i + 1] + smoothed_path[j:]
    
    return smoothed_path

def plot_path_with_boundaries_and_mixed_obstacles(path, walls=None, goal_area=None, outside_walls=None):
    
    # Plot 2D walls as boxes if provided
    plt.figure(figsize=(8, 6))
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

    # Plot the path
    path = np.array(path)  # Ensure path is a numpy array
    plt.plot(path[:, 0], path[:, 1], 'bo-', label='Path')

    # Plot the start and goal positions
    plt.plot(path[0, 0], path[0, 1], 'go', label='Start', markersize=10)
    plt.plot(path[-1, 0], path[-1, 1], 'ro', label='Goal', markersize=10)

    # Set limits
    plt.xlim(-0.6, 1.6)
    plt.ylim(-0.5, 0.5)

    # Add labels and title
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.title("Kinodynamic RRT Path with Map Boundaries and Obstacles")
    plt.legend()

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

def move_ball_along_path_with_pid(model, data, path, window, scene, context, options, viewport, camera, pid_x, pid_y):
    
    ball_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ball")
    dt = 0.01
    fig, ax, line_dev = initialize_plot()
    time_data, deviation_data = [], []
    time_elapsed = 0
    max_speed, min_speed = 10, 1
    slowdown_distance = 1.25

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
            if distance_to_goal < 0.05:
                print("Reached node", i + 1)
                break

            desired_speed = update_speed(distance_to_goal, slowdown_distance, max_speed, min_speed)
            print(f"Distance to goal: {distance_to_goal}, Desired speed: {desired_speed}")
            
            control_x, control_y = apply_pid_control(pid_x, pid_y, ball_pos, dt, desired_speed)
            print(f"Control signals: control_x={control_x}, control_y={control_y}")

            data.ctrl[0], data.ctrl[1] = control_x, control_y
            mujoco.mj_step(model, data)
            render_scene(model, data, options, scene, context, viewport, camera, window)

            deviation = calculate_deviation(start_pos, end_pos, ball_pos)
            deviation_data.append(deviation)
            time_data.append(time_elapsed)
            line_dev.set_xdata(time_data)
            line_dev.set_ydata(deviation_data)
            ax.set_xlim(0, max(10, time_elapsed + 1))
            fig.canvas.draw()
            fig.canvas.flush_events()
            time_elapsed += dt
    
    plt.close() 
    return time_elapsed

def planning_time(pid_x, pid_y, model, data, path):
    
    ball_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ball")  # Get ball body ID
    dt = 0.01  # Time step
    time_elapsed = 0
    max_speed = 8  # Maximum speed of the ball
    min_speed = 1   # Minimum speed when close to the point
    slowdown_distance = 1.25  # Distance at which to start slowing down

    # Iterate over the path segments (line between each pair of nodes)
    for i in range(len(path) - 1):
        
        start_pos = np.array(path[i])
        end_pos = np.array(path[i + 1])
        line_direction = end_pos - start_pos
        line_length = np.linalg.norm(line_direction)
        line_direction_normalized = line_direction / line_length

        while True:
            ball_pos = data.xpos[ball_id][:2]  # Get current ball position (x, y)
            
            # Compute vector to the end position
            ball_to_end = end_pos - ball_pos
            distance_to_goal = np.linalg.norm(ball_to_end)

            # If the ball is very close to the end of the segment, move to the next segment
            if distance_to_goal < 0.05:
                print("Reached node", i + 1)
                break

            # Dynamically calculate the desired speed based on the distance to the next point
            if distance_to_goal < slowdown_distance:
                desired_speed = max(min_speed, (distance_to_goal / slowdown_distance) * max_speed)
            else:
                desired_speed = max_speed

            # Adjust PID controller gains dynamically for x and y directions
            control_x = pid_x.compute(ball_pos[0], dt)
            control_y = pid_y.compute(ball_pos[1], dt)

            # Normalize the control signals to match the desired speed
            control_vector = np.array([control_x, control_y])
            control_magnitude = np.linalg.norm(control_vector)
            
            if control_magnitude > 0:
                control_vector_normalized = control_vector / control_magnitude
                control_x = control_vector_normalized[0] * desired_speed
                control_y = control_vector_normalized[1] * desired_speed

            # Apply control to the ball's actuators
            data.ctrl[0] = control_x  # x direction control
            data.ctrl[1] = control_y  # y direction control
            
            mujoco.mj_step(model, data)

            # Update time elapsed
            time_elapsed += dt
            
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
    pid_x = PIDController(kp=.45, ki=0.0, kd=0.5)
    pid_y = PIDController(kp=.45, ki=0.0, kd=0.5)
    path, tree = kinodynamic_rrt(pid_x, pid_y, start_pos, goal_area, walls, outside_walls=outside_walls)
    
    if path:
        
        #path = smooth_path(path, walls)  # Smooth the path
        plot_path_with_boundaries_and_mixed_obstacles(path, walls, goal_area, outside_walls)
        
        pid_x = PIDController(kp=.6, ki=0.0, kd=0.5)
        pid_y = PIDController(kp=.6, ki=0.0, kd=0.5)
        
        # Initialize the window and visualization structures
        window, camera, scene, context, options, viewport = init_glfw_window(model)
        
        # Control the ball to follow the path
        move_ball_along_path_with_pid(model, data, path, window, scene, context, options, viewport, camera, pid_x, pid_y)
    
    else:
        print("No path found")

    plt.close('all')
    glfw.terminate()
    
    return

def tree_visualization(start_pos, walls, goal_area, outside_walls):
    
    pid_x = PIDController(kp=.45, ki=0.0, kd=0.5)
    pid_y = PIDController(kp=.45, ki=0.0, kd=0.5)
    
    for trial in range(5):
        seed = trial
        random.seed(seed)
        np.random.seed(seed)
        path, tree = kinodynamic_rrt(pid_x, pid_y, start_pos, goal_area, walls)
        print(f"Trial {trial + 1} - Path: {'Found' if path else 'Not Found'}")
        visualize_tree(tree, path, goal_area, walls, outside_walls, start_pos)

    return

def run_trials(start_pos, goal_area, walls, outside_walls, num_trials, Tmax):
    
    success_count = 0
    plan_time = -1
    
    for trial in range(num_trials):
        print(f"Running trial {trial + 1}/{num_trials}...")

        # Load the MuJoCo model
        model = mujoco.MjModel.from_xml_path("ball_square.xml")
        data = mujoco.MjData(model)
        
        pid_x = PIDController(kp=.45, ki=0.0, kd=0.5)
        pid_y = PIDController(kp=.45, ki=0.0, kd=0.5)

        # Generate a path using kinodynamic RRT
        path, tree = kinodynamic_rrt(pid_x, pid_y, start_pos, goal_area, walls)
        
        if path:
            # Smooth the path to avoid obstacles
            path = smooth_path(path, walls)
            
            # Calculate planning time for the smoothed path
            plan_time = planning_time(pid_x, pid_y, model, data, path)
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
                tree_visualization(pid_x, pid_y, start_pos, walls, goal_area, outside_walls)
            elif choice == 3:
                num_trials = 10
                print("start_pos:", start_pos)
                
                Tmax_input = input("Enter Tmax: ")
                if Tmax_input.isdigit():
                    Tmax = int(Tmax_input)
                    if Tmax <= 0:
                        print("Please enter a positive integer for Tmax.")
                        continue
                else:
                    print("Invalid input. Please enter a positive integer.")
                    continue
                
                run_trials(pid_x, pid_y, start_pos, goal_area, walls, outside_walls, num_trials, Tmax)
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
