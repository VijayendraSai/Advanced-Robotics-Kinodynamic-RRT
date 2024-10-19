import mujoco
import glfw
import numpy as np
import random
import matplotlib.pyplot as plt

class Node:
    def __init__(self, position, parent=None):
        self.position = position
        self.parent = parent
        self.control = None

def kinodynamic_rrt(start_pos, goal_pos, walls, N=100000):
    
    T = [Node(start_pos)]  # Initialize the tree with the start node
    accepted_distance = 0.1
    
    for _ in range(N):
        xrand = sample_random_position()  # Sample a random position in the environment
        xnear = nearest(T, xrand)         # Find the nearest node in the tree
        ue = choose_control(xnear.position, xrand)  # Choose control input
        xe = simulate(xnear.position, ue)  # Simulate the new position
        
        # Updated to pass walls parameter to is_collision_free
        if is_collision_free(xe, walls):
            new_node = Node(xe, parent=xnear)
            new_node.control = ue
            T.append(new_node)    
            if np.linalg.norm(np.array(goal_pos) - np.array(xe)) < accepted_distance:  # Check if xe is close to goal
                return construct_path(new_node)
    
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

def move_ball_to_position(model, data, target_pos, window, scene, context, options, viewport, camera):
    
    ball_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ball")  # Get ball body ID
    ramp_up_factor = 0.2  # Start ramp-up factor higher for quicker initial speed
    max_speed = 1  # Maximum allowed speed to prevent overshooting

    while True:
        ball_pos = data.xpos[ball_id][:2]  # Get ball position (x, y)

        # Compute the direction to move in
        direction = np.array(target_pos[:2]) - ball_pos
        distance_to_target = np.linalg.norm(direction)

        # Slowly ramp up the velocity as the ball moves from the checkpoint
        ramp_up_factor = min(ramp_up_factor + 0.05, 1.0)  # Increase ramp-up factor faster

        # Slow down the ball as it approaches the target by scaling the velocity with distance
        slowing_factor = min(distance_to_target, 1)  # Slow down when closer to the target (faster when far)
        
        # Proportional control with ramp-up and slowing factor
        ball_vel = direction * slowing_factor * ramp_up_factor * 2  # Increase velocity scaling factor
        
        # Clamp the velocity to max speed to avoid overshooting
        ball_vel = np.clip(ball_vel, -max_speed, max_speed)

        # Apply control to the ball's actuators
        data.ctrl[0] = ball_vel[0]  # x direction control
        data.ctrl[1] = ball_vel[1]  # y direction control
        
        mujoco.mj_step(model, data)

        # Render the scene
        mujoco.mjv_updateScene(model, data, options, None, camera, mujoco.mjtCatBit.mjCAT_ALL, scene)
        mujoco.mjr_render(viewport, scene, context)

        # Check for glfw window events
        glfw.poll_events()

        # Swap the front and back buffers
        glfw.swap_buffers(window)

        print(f'pos: {ball_pos}, speed: {ball_vel}, distance_to_target: {distance_to_target}, ramp_up_factor: {ramp_up_factor}')

        # Apply a stronger brake when very close to the target
        if distance_to_target < 0.1:
            ramp_up_factor = 0.1  # Reduce ramp-up to stop acceleration near the target

        # Stop if ball is close enough to the target position
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

# Using the RRT in the main function
if __name__ == "__main__":
    
    # Load the MuJoCo model
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

    # Perform Kinodynamic-RRT to find a path
    path = kinodynamic_rrt(start_pos, goal_pos, walls)
    print(path)

    if path:
        plot_path_with_boundaries_and_mixed_obstacles(path, walls, goal_area, outside_walls)
        # Initialize the window and visualization structures
        window, camera, scene, context, options, viewport = init_glfw_window(model)
        print(f"Path found: {path}")
        # Control the ball to follow the path
        for target_pos in path:
            move_ball_to_position(model, data, target_pos, window, scene, context, options, viewport, camera)
    else:
        print("No path found")

    # Close the window and terminate glfw
    glfw.terminate()
