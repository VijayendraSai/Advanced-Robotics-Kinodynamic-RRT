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
        # Compute error between current position and setpoint
        error = self.setpoint - current_value
        
        # Proportional term
        p_term = self.kp * error
        
        # Integral term (summing errors over time)
        self.integral += error * dt
        i_term = self.ki * self.integral
        
        # Derivative term (rate of change of error)
        d_term = self.kd * (error - self.previous_error) / dt
        self.previous_error = error
        
        # Combine the terms to compute the control output
        return p_term + i_term + d_term

class Node:
    def __init__(self, position, parent=None):
        self.position = position
        self.parent = parent
        self.control = None

def kinodynamic_rrt(start_pos, goal_pos, walls, seed=0, N=100000):
    random.seed(seed)
    np.random.seed(seed)
    
    T = [Node(start_pos)]  # Initialize the tree with the start node
    accepted_distance = 0.1
    
    for _ in range(N):
        xrand = sample_random_position()  # Sample a random position in the environment
        xnear = nearest(T, xrand)         # Find the nearest node in the tree
        ue = choose_control(xnear.position, xrand)  # Choose control input
        xe = simulate(xnear.position, ue)  # Simulate the new position
        
        if is_collision_free(xe, walls):
            new_node = Node(xe, parent=xnear)
            new_node.control = ue
            T.append(new_node)    
            if np.linalg.norm(np.array(goal_pos) - np.array(xe)) < accepted_distance:
                return construct_path(new_node), T
    
    return None, T  # Return None if no path is found, along with the tree

def sample_random_position():
    return np.array([random.uniform(-0.5, 1.5), random.uniform(-0.4, 0.4)])  # Adjust to map bounds

def nearest(T, xrand):
    return min(T, key=lambda node: np.linalg.norm(node.position - xrand))

def choose_control(xnear, xrand):
    direction = np.array(xrand) - np.array(xnear)
    return direction / np.linalg.norm(direction)  # Normalize to get direction of control

def simulate(xnear, control, dt=0.05):
    return np.array(xnear) + np.array(control) * dt

def is_collision_free(xe, walls, safety_margin=.15):
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

def construct_path(node):
    # Reconstruct the path from the goal node to the start node
    path = []
    while node is not None:
        path.append(node.position)
        node = node.parent
    return path[::-1]  # Return the path from start to goal

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

def move_ball_to_position_with_pid(model, data, target_pos, window, scene, context, options, viewport, camera, pid_x, pid_y):
    
    ball_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ball")  # Get ball body ID
    dt = 0.01  # Simulation timestep

    while True:
        ball_pos = data.xpos[ball_id][:2]  # Get current ball position (x, y)
        distance_to_target = np.linalg.norm(np.array(target_pos[:2]) - ball_pos)

        # Update PID controllers for x and y directions
        pid_x.setpoint = target_pos[0]
        pid_y.setpoint = target_pos[1]
        
        control_x = pid_x.compute(ball_pos[0], dt)
        control_y = pid_y.compute(ball_pos[1], dt)
        
        # Clamp the control signal to prevent overshooting
        max_speed = 1.0
        control_x = np.clip(control_x, -max_speed, max_speed)
        control_y = np.clip(control_y, -max_speed, max_speed)

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

        print(f'pos: {ball_pos}, control_x: {control_x}, control_y: {control_y}, distance_to_target: {distance_to_target}')

        # Stop if the ball is close enough to the target position
        if distance_to_target < 0.05:
            print("Reached checkpoint")
            break

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

# Using the RRT in the main function
if __name__ == "__main__":
    
    # Load the MuJoCo model
    # model = mujoco.MjModel.from_xml_path("ball_square.xml")
    model = mujoco.MjModel.from_xml_path("ball_square.xml")
    data = mujoco.MjData(model)
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

    # Define the start and goal positions
    start_pos = [0, 0]  # Starting at the origin
    goal_pos = [0.9, 0]  # Goal position based on XML map

    # Perform and visualize 5 different trials of the Kinodynamic-RRT
    for trial in range(5):
        seed = trial  # Use different seed for each trial
        path, tree = kinodynamic_rrt(start_pos, goal_pos, walls, seed=seed)
        print(f"Trial {trial + 1} - Path: {'Found' if path else 'Not Found'}")
        visualize_tree(tree, goal_pos, walls, goal_area, outside_walls)

    # # Perform Kinodynamic-RRT to find a path
    # path = kinodynamic_rrt(start_pos, goal_pos, walls)
    # print(path)

    # if path:
    #     plot_path_with_boundaries_and_mixed_obstacles(path, walls, goal_area, outside_walls)
    #     # Initialize the window and visualization structures
    #     window, camera, scene, context, options, viewport = init_glfw_window(model)
        
    #     print(f"Path found: {path}")
    #     # Control the ball to follow the path
    #     # Create PID controllers for x and y coordinates
    #     pid_x = PIDController(kp=0.1, ki=0.0, kd=0.40)  # Lower kp, higher kd
    #     pid_y = PIDController(kp=0.1, ki=0.0, kd=0.40)

    #     for target_pos in path:
    #         move_ball_to_position_with_pid(model, data, target_pos, window, scene, context, options, viewport, camera, pid_x, pid_y)
    # else:
    #     print("No path found")

    # Close the window and terminate glfw
    glfw.terminate()
