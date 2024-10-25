import mujoco
import glfw
import numpy as np
import random
import matplotlib.pyplot as plt

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


def kinodynamic_rrt(start_pos, goal_area, walls, N=1000):
    
    T = [Node(start_pos)]  # Initialize the tree with the start node
    accepted_distance = 0.10  # You can remove this if goal_area is used

    for _ in range(N):
        xrand = sample_random_position()  # Sample a random position in the environment
        xnear = nearest(T, xrand)         # Find the nearest node in the tree
        ue = choose_control(xnear.position, xrand)  # Choose control input
        xe = simulate(xnear.position, ue)  # Simulate the new position
        
        # Check if the new position is collision-free
        if is_collision_free(xe, walls):
            new_node = Node(xe, parent=xnear)
            new_node.control = ue
            T.append(new_node)
            
            # Check if the new position is inside the goal area
            if is_in_goal_area(xe, goal_area):
                print("Reached goal area")
                return construct_path(new_node)  # Construct and return the path

    return None  # Return None if no path is found

def sample_random_position():
    return np.array([random.uniform(-0.5, 1.5), random.uniform(-0.4, 0.4)])  # Adjust to map bounds

def nearest(T, xrand):
    return min(T, key=lambda node: np.linalg.norm(node.position - xrand))

def choose_control(xnear, xrand):
    direction = np.array(xrand) - np.array(xnear)
    return direction / np.linalg.norm(direction)  # Normalize to get direction of control

def simulate(xnear, control, dt=0.05):
    return np.array(xnear) + np.array(control) * dt

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

def move_ball_along_path_with_pid(model, data, path, window, scene, context, options, viewport, camera, pid_x, pid_y):
    ball_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ball")  # Get ball body ID
    
    dt = 0.01  # Time step
    
    # Initialize live plot for distance from line
    plt.ion()  # Enable interactive mode
    fig, ax = plt.subplots()
    time_data = []
    line_dev, = ax.plot([], [], 'b-', label='Deviation')
    deviation_data = []  # To store deviation data
    ax.set_xlim(0, 10)  # Set reasonable x limits (time)
    ax.set_ylim(-0.5, 0.5)  # Set y limits for deviations
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Deviation from Line (m)')
    ax.legend()
    plt.grid(True)

    time_elapsed = 0
    max_speed = 10  # Maximum speed of the ball
    min_speed = 1   # Minimum speed when close to the point
    slowdown_distance = 1.0  # Distance at which to start slowing down

    # Iterate over the path segments (line between each pair of nodes)
    for i in range(len(path) - 1):
        
        start_pos = np.array(path[i])
        end_pos = np.array(path[i + 1])
        line_direction = end_pos - start_pos
        line_length = np.linalg.norm(line_direction)
        line_direction_normalized = line_direction / line_length
        

        # Update the PID controller setpoints to the next node's position
        pid_x.setpoint = end_pos[0]
        pid_y.setpoint = end_pos[1]

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

            # Print debugging information
            print(f"Distance to goal: {distance_to_goal}, Desired speed: {desired_speed}")

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

            # Print control signals for debugging
            print(f"Control signals: control_x={control_x}, control_y={control_y}")

            # Apply control to the ball's actuators
            data.ctrl[0] = control_x  # x direction control
            data.ctrl[1] = control_y  # y direction control
            
            mujoco.mj_step(model, data)

            # Render the scene
            mujoco.mjv_updateScene(model, data, options, None, camera, mujoco.mjtCatBit.mjCAT_ALL, scene)
            mujoco.mjr_render(viewport, scene, context)

            # Check for glfw window events
            glfw.poll_events()

            # Swap the front and back buffers
            glfw.swap_buffers(window)
            
            # Compute deviation from the path
            A = start_pos
            B = end_pos
            P = ball_pos
            deviation = (B[0] - A[0]) * (A[1] - P[1]) - (A[0] - P[0]) * (B[1] - A[1]) / np.sqrt((B[0] - A[0]) ** 2 + (B[1] - A[1]) ** 2)
            
            # Store and plot deviation
            deviation_data.append(deviation)
            time_data.append(time_elapsed)

            line_dev.set_xdata(time_data)
            line_dev.set_ydata(deviation_data)
            
            ax.set_xlim(0, max(10, time_elapsed + 1))  # Dynamically adjust the x-axis limit
            fig.canvas.draw()
            fig.canvas.flush_events()

            time_elapsed += dt

    return time_elapsed

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

def planning_time(model, data, path):
    ball_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ball")  # Get ball body ID
    
    dt = 0.01  # Time step
    time_elapsed = 0
    max_speed = 10  # Maximum speed of the ball
    min_speed = 1   # Minimum speed when close to the point
    slowdown_distance = 1.0  # Distance at which to start slowing down

    # Iterate over the path segments (line between each pair of nodes)
    for i in range(len(path) - 1):
        
        start_pos = np.array(path[i])
        end_pos = np.array(path[i + 1])
        line_direction = end_pos - start_pos
        line_length = np.linalg.norm(line_direction)
        line_direction_normalized = line_direction / line_length

        # Initialize the PID controllers
        pid_x = PIDController(kp=1.0, ki=0.0, kd=0.1, setpoint=end_pos[0])
        pid_y = PIDController(kp=1.0, ki=0.0, kd=0.1, setpoint=end_pos[1])

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

def visualize_tree(T, goal_pos, walls=None, goal_area=None, outside_walls=None):
    plt.figure(figsize=(8, 6))

    # Plot walls as boxes
    if walls:
        for wall, coordinates in walls.items():
            wall_polygon = plt.Polygon(coordinates, color='red', alpha=0.5)
            plt.gca().add_patch(wall_polygon)

    # Plot outside walls as lines
    if outside_walls:
        for wall in outside_walls:
            plt.plot([wall[0][0], wall[1][0]], [wall[0][1], wall[1][1]], 'k-', lw=2)

    # Plot the goal area as a box
    if goal_area:
        goal_polygon = plt.Polygon(goal_area, color='green', alpha=0.3)
        plt.gca().add_patch(goal_polygon)

    # Plot the nodes and edges of the tree
    for node in T:
        if node.parent:
            plt.plot([node.position[0], node.parent.position[0]], [node.position[1], node.parent.position[1]], 'b-', alpha=0.5)
        plt.plot(node.position[0], node.position[1], 'bo', markersize=3)

    # Plot the start and goal positions
    start_pos = T[0].position
    plt.plot(start_pos[0], start_pos[1], 'go', label='Start', markersize=10)
    plt.plot(goal_pos[0], goal_pos[1], 'ro', label='Goal', markersize=10)

    # Set limits
    plt.xlim(-0.6, 1.6)
    plt.ylim(-0.5, 0.5)

    # Add labels and title
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.title("Kinodynamic-RRT Tree")
    plt.legend()

    plt.grid(True)
    plt.show()

def model_creation(start_pos, goal_area, walls, outside_walls):
    # Load the MuJoCo model
    model = mujoco.MjModel.from_xml_path("ball_square.xml")
    data = mujoco.MjData(model)
    
    # Perform Kinodynamic-RRT to find a path that reaches the goal area
    path = kinodynamic_rrt(start_pos, goal_area, walls)
    
    if path:
        path = smooth_path(path, walls)  # Smooth the path
        plot_path_with_boundaries_and_mixed_obstacles(path, walls, goal_area, outside_walls)
        
        # Initialize the window and visualization structures
        window, camera, scene, context, options, viewport = init_glfw_window(model)
        
        # Create PID controllers for x and y coordinates
        pid_x = PIDController(kp=.45, ki=0.0, kd=0.5)  # Lower kp, higher kd
        pid_y = PIDController(kp=.45, ki=0.0, kd=0.5)
        
        # Control the ball to follow the path
        move_ball_along_path_with_pid(model, data, path, window, scene, context, options, viewport, camera, pid_x, pid_y)
    else:
        print("No path found")
    
    # Close the window and terminate glfw
    glfw.terminate()

def tree_visualization(start_pos, walls, goal_area, outside_walls):
    for trial in range(5):
        seed = trial
        path, tree = kinodynamic_rrt(start_pos, walls, seed=seed)
        print(f"Trial {trial + 1} - Path: {'Found' if path else 'Not Found'}")
        visualize_tree(tree, walls, goal_area, outside_walls)

def run_trials(start_pos, goal_area, walls, outside_walls, num_trials, Tmax):
    success_count = 0
    plan_time = -1
    
    for trial in range(num_trials):
        print(f"Running trial {trial + 1}/{num_trials}...")

        # Load the MuJoCo model
        model = mujoco.MjModel.from_xml_path("ball_square.xml")
        data = mujoco.MjData(model)

        # Generate a path using kinodynamic RRT
        path = kinodynamic_rrt(start_pos, goal_area, walls)
        
        if path:
            # Smooth the path to avoid obstacles
            path = smooth_path(path, walls)

            # Plot the path along with boundaries and obstacles
            plot_path_with_boundaries_and_mixed_obstacles(path, walls, goal_area, outside_walls)
            
            # # Initialize the window and visualization structures
            # window, camera, scene, context, options, viewport = init_glfw_window(model)
            
            # # Create PID controllers for x and y coordinates
            # pid_x = PIDController(kp=.45, ki=0.0, kd=0.5)  # Lower kp, higher kd
            # pid_y = PIDController(kp=.45, ki=0.0, kd=0.5)
            
            # # Control the ball to follow the path
            #temp = move_ball_along_path_with_pid(model, data, path, window, scene, context, options, viewport, camera, pid_x, pid_y)
            #print("temp: ", temp)

            # # Close the window and terminate glfw
            #glfw.terminate()

            # Calculate planning time for the smoothed path
            plan_time = planning_time(model, data, path)
            glfw.terminate()
            #plan_time = temp
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

    
    # print(f"Planning time: {plan_time:.2f} seconds")

def main():
    # Define the goal area (a rectangular region)
    goal_area = [[0.9, -0.3], [0.9, 0.3], [1.1, 0.3], [1.1, -0.3]]
    
    # Define outside walls as lines
    outside_walls = [
        [[-0.5, -0.4], [-0.5, 0.4]],
        [[1.5, -0.4], [1.5, 0.4]],
        [[-0.5, 0.4], [1.5, 0.4]],
        [[-0.5, -0.4], [1.5, -0.4]]]

    # Define the middle obstacle
    walls = {
        "wall_3": [[0.5, -0.15], [0.5, 0.15], [0.6, 0.15], [0.6, -0.15]]}

    # Define the start position
    start_pos = [0, 0]  # Starting at the origin
    
    while True:
        print("\nMenu:")
        print("1. Model Creation")
        print("2. Tree Visualization")
        print("3. Planning Time")
        print("4. Quit")        
        choice = int(input("Enter your choice: "))      
        if choice == 1:
            model_creation(start_pos, goal_area, walls, outside_walls)
        elif choice == 2:
            tree_visualization(start_pos, walls, goal_area, outside_walls)
            # Close the window and terminate glfw
            glfw.terminate()
        elif choice == 3:
            num_trials = 10
            print("start_pos: ", start_pos)
            Tmax = int(input("Enter Tmax: "))
            run_trials(start_pos, goal_area, walls, outside_walls, num_trials, Tmax)
        elif choice == 4:
            print("Exiting the program. Goodbye!")
            break
        else:
            print("Invalid option. Please try again.")

if __name__ == "__main__":
    main()