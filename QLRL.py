import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import pdist, squareform
import random
from tqdm import tqdm
from math import radians, sin, cos, sqrt, asin
import time
import os

# Create output directory for saving visualizations
os.makedirs('tsp_visualizations', exist_ok=True)

# Function to calculate Haversine distance (in km) between two points given by latitude/longitude
def haversine_distance(lat1, lon1, lat2, lon2):
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371  # Radius of Earth in kilometers
    return c * r

# Load dataset
file_path = "/Users/architmahajan/Desktop/ResearchPapercode/Indian Cities Database.csv"
df = pd.read_csv(file_path)

# Select cities for TSP
n_cities = len(df)
city_names = df['City'].tolist()

# Display cities and allow user to select starting city
print("Available cities:")
for i, city in enumerate(city_names):
    print(f"{i}: {city}")

# Get user input for starting city
while True:
    try:
        START_CITY = int(input(f"Enter the index of your starting city (0-{n_cities-1}): "))
        if 0 <= START_CITY < n_cities:
            print(f"Selected starting city: {city_names[START_CITY]}")
            break
        else:
            print(f"Please enter a valid index between 0 and {n_cities-1}")
    except ValueError:
        print("Please enter a valid integer")

# Create base distance matrix using Haversine distance
base_distance_matrix = np.zeros((n_cities, n_cities))
for i in range(n_cities):
    for j in range(n_cities):
        if i != j:
            base_distance_matrix[i, j] = haversine_distance(
                df.iloc[i]['Lat'], df.iloc[i]['Long'],
                df.iloc[j]['Lat'], df.iloc[j]['Long']
            )

# Traffic simulation parameters
TRAFFIC_UPDATE_FREQUENCY = 1000  # Update traffic every n episodes
HIGH_TRAFFIC_MULTIPLIER_MAX = 2.5  # Maximum traffic multiplier
MEDIUM_TRAFFIC_MULTIPLIER_MAX = 1.5  # Medium traffic multiplier
LOW_TRAFFIC_MULTIPLIER_MAX = 1.2  # Low traffic multiplier
TRAFFIC_PATTERNS = ['peak_hours', 'normal', 'low']  # Different traffic patterns

# Traffic pattern time dependency (simulated time of day effect)
def get_time_based_traffic_pattern():
    """Simulate time of day effects on traffic patterns"""
    # Get current hour (0-23) for simulation - here we just use a simple calculation
    current_hour = (int(time.time() / 10) % 24)  # Changes every 10 seconds for simulation
    
    # Morning and evening rush hours
    if current_hour in [7, 8, 9, 17, 18, 19]:
        return 'peak_hours'
    elif current_hour in [10, 11, 12, 13, 14, 15, 16]:
        return 'normal'
    else:
        return 'low'

# Function to update distance matrix based on traffic conditions
def update_distance_matrix(base_matrix):
    """
    Update the distance matrix based on simulated traffic conditions
    Returns the adjusted matrix and a traffic heatmap for visualization
    """
    # Get current traffic pattern
    current_pattern = get_time_based_traffic_pattern()
    
    # Create a copy of the base matrix to adjust
    adjusted_matrix = base_matrix.copy()
    traffic_heatmap = np.zeros((n_cities, n_cities))
    
    # Adjust distances based on traffic pattern
    for i in range(n_cities):
        for j in range(n_cities):
            if i != j:
                # Population-based traffic probability (larger cities have more traffic)
                city_size_i = min(len(df.iloc[i]['State']), 10) / 10  # Proxy for city size
                city_size_j = min(len(df.iloc[j]['State']), 10) / 10
                
                # Base probability of traffic between these cities
                traffic_prob = (city_size_i + city_size_j) / 2
                
                # Adjust based on pattern
                if current_pattern == 'peak_hours':
                    traffic_multiplier = random.uniform(1.0, HIGH_TRAFFIC_MULTIPLIER_MAX) if random.random() < (0.6 + traffic_prob) else 1.0
                elif current_pattern == 'normal':
                    traffic_multiplier = random.uniform(1.0, MEDIUM_TRAFFIC_MULTIPLIER_MAX) if random.random() < (0.3 + traffic_prob) else 1.0
                else:  # low traffic
                    traffic_multiplier = random.uniform(1.0, LOW_TRAFFIC_MULTIPLIER_MAX) if random.random() < (0.1 + traffic_prob) else 1.0
                
                # Apply traffic multiplier to distance
                adjusted_matrix[i, j] = base_matrix[i, j] * traffic_multiplier
                traffic_heatmap[i, j] = traffic_multiplier  # Store actual multiplier for visualization
    
    return adjusted_matrix, traffic_heatmap, current_pattern

# Hyperparameters
EPSILON_START = 0.9
EPSILON_END = 0.01
ALPHA = 0.1  # Learning rate
GAMMA = 0.9  # Discount factor
NUM_EPISODES = 50000

# Function to calculate distances for a given path
def calculate_path_distances(path, distance_matrix):
    total_distance = 0
    for i in range(len(path) - 1):
        city1, city2 = path[i], path[i+1]
        total_distance += distance_matrix[city1, city2]
    return total_distance

# Function to train the Q-Learning model
def train_q_learning(distance_matrix, use_traffic=True, description="Q-Learning"):
    # Initialize Q-table
    Q_table = np.zeros((n_cities, n_cities))
    
    # Training with epsilon decay and progress bar
    epsilon_decay = (EPSILON_START - EPSILON_END) / NUM_EPISODES
    epsilon = EPSILON_START
    
    # Track best path and its distance for convergence monitoring
    best_distance = float('inf')
    best_path = []
    convergence_data = []
    traffic_history = []  # Track traffic patterns over time
    
    # Current distance matrix (may be updated with traffic)
    current_distance_matrix = distance_matrix.copy()
    traffic_heatmap = np.ones((n_cities, n_cities))  # Default no traffic
    current_pattern = "none" if not use_traffic else "normal"
    
    # Start timing the training process
    training_start_time = time.time()
    
    # Training loop
    for episode in tqdm(range(NUM_EPISODES), desc=f"Training {description}"):
        # Update traffic conditions periodically if we're using traffic
        if use_traffic and episode % TRAFFIC_UPDATE_FREQUENCY == 0:
            current_distance_matrix, traffic_heatmap, current_pattern = update_distance_matrix(distance_matrix)
            traffic_history.append((episode, current_pattern))
        
        current_city = START_CITY
        visited = set([current_city])
        path = [current_city]
        total_distance = 0
        
        # Build route until all cities are visited
        while len(visited) < n_cities:
            # Available cities (not yet visited)
            available_cities = [c for c in range(n_cities) if c not in visited]
            
            # Exploration vs exploitation
            if random.uniform(0, 1) < epsilon:
                next_city = random.choice(available_cities)  # Explore
            else:
                # Get Q-values for available cities only
                available_q_values = [Q_table[current_city, c] if c not in visited else -np.inf 
                                    for c in range(n_cities)]
                next_city = np.argmax(available_q_values)  # Exploit
                
                # Fallback if next_city is already visited
                if next_city in visited:
                    next_city = random.choice(available_cities)
            
            # Calculate reward (negative distance with traffic consideration)
            reward = -current_distance_matrix[current_city, next_city]
            total_distance -= reward  # Track actual distance
            
            # Update Q-table with traffic-aware values
            future_value = max([Q_table[next_city, c] for c in range(n_cities) if c not in visited and c != next_city], default=0)
            Q_table[current_city, next_city] += ALPHA * (reward + GAMMA * future_value - Q_table[current_city, next_city])
            
            # Move to next city
            current_city = next_city
            visited.add(current_city)
            path.append(current_city)
        
        # Return to start city to complete the tour
        reward = -current_distance_matrix[current_city, START_CITY]
        total_distance -= reward
        
        # Update Q-value for return to start
        Q_table[current_city, START_CITY] += ALPHA * (reward - Q_table[current_city, START_CITY])
        
        # Check if this is the best path so far
        if total_distance < best_distance:
            best_distance = total_distance
            best_path = path + [START_CITY]  # Add return to start
            
        # Record progress for analysis
        if episode % 100 == 0:
            convergence_data.append((episode, best_distance))
        
        # Decay epsilon
        epsilon = max(EPSILON_END, epsilon - epsilon_decay)
    
    # End timing the training process
    training_end_time = time.time()
    training_time = training_end_time - training_start_time
    print(f"\n{description} training completed in {training_time:.2f} seconds")
    
    return Q_table, best_path, best_distance, convergence_data, traffic_history, training_time

# Generate an optimal path using Q-values
def get_optimal_path(Q_table, distance_matrix, use_traffic=True):
    path_calc_start_time = time.time()
    
    # Update traffic one final time for the optimal route if using traffic
    if use_traffic:
        current_distance_matrix, current_traffic, pattern = update_distance_matrix(distance_matrix)
    else:
        current_distance_matrix = distance_matrix.copy()
        current_traffic = np.ones((n_cities, n_cities))  # No traffic multiplier
        pattern = "none"
    
    path = [START_CITY]
    current_city = START_CITY
    visited = set([current_city])
    
    while len(visited) < n_cities:
        available_cities = [c for c in range(n_cities) if c not in visited]
        
        # Use both Q-values and current traffic conditions for decision
        # Higher weight to Q-values (learned knowledge) but also consider current traffic
        decision_values = {}
        for city in available_cities:
            q_value_weight = 0.7  # Weight for Q-value
            traffic_weight = 0.3  # Weight for current traffic
            
            # Normalize Q-value
            q_value = Q_table[current_city, city]
            
            # Consider inverse of traffic-adjusted distance (lower is better)
            traffic_factor = 1.0 / max(0.1, current_distance_matrix[current_city, city])
            
            # Combined decision value
            decision_values[city] = (q_value * q_value_weight) + (traffic_factor * traffic_weight)
        
        # Choose the city with the best combined value
        next_city = max(decision_values.items(), key=lambda x: x[1])[0]
        
        path.append(next_city)
        visited.add(next_city)
        current_city = next_city
    
    # Return to starting city to complete the tour
    path.append(START_CITY)
    
    # Calculate distances
    path_distance = calculate_path_distances(path, current_distance_matrix)
    
    # End timing the optimal path calculation
    path_calc_end_time = time.time()
    path_calc_time = path_calc_end_time - path_calc_start_time
    
    return path, path_distance, current_traffic, pattern, path_calc_time

# Function to plot convergence
def plot_convergence(convergence_data, traffic_history=None, title="Convergence", filename="convergence.png"):
    plt.figure(figsize=(12, 8))
    
    # Plot convergence
    episodes, distances = zip(*convergence_data)
    plt.plot(episodes, distances, 'b-', linewidth=2)
    plt.xlabel("Episode", fontsize=14)
    plt.ylabel("Best Tour Distance (km)", fontsize=14)
    plt.title(title, fontsize=16)
    plt.grid(True)
    
    # If we have traffic history, overlay traffic pattern changes
    if traffic_history:
        ax2 = plt.gca().twinx()
        pattern_colors = {'peak_hours': 'red', 'normal': 'orange', 'low': 'green'}
        
        for i, (episode, pattern) in enumerate(traffic_history):
            ax2.axvline(x=episode, color=pattern_colors[pattern], alpha=0.2, linewidth=1)
        
        ax2.set_yticks([])
        ax2.set_ylabel("Traffic Patterns", fontsize=14)
        
        # Add a legend for traffic patterns
        from matplotlib.lines import Line2D
        pattern_legend_elements = [Line2D([0], [0], color=color, lw=2, label=pattern) 
                                  for pattern, color in pattern_colors.items()]
        ax2.legend(handles=pattern_legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(os.path.join('tsp_visualizations', filename), dpi=300)
    plt.close()

# Function to plot route on map
def plot_route_map(path, traffic_matrix=None, title="TSP Route", filename="route_map.png", distance=None, travel_time=None):
    plt.figure(figsize=(14, 10))
    
    # Plot all cities
    plt.scatter(df['Long'], df['Lat'], c='blue', s=50, label='Cities')
    
    # Extract coordinates for the route
    route_lats = [df.iloc[city_idx]['Lat'] for city_idx in path]
    route_longs = [df.iloc[city_idx]['Long'] for city_idx in path]
    
    # If no traffic data, just plot the basic route
    if traffic_matrix is None or np.all(traffic_matrix == 1):
        plt.plot(route_longs, route_lats, 'r-', linewidth=2.5, 
                 label=f'Route: {distance:.2f} km')
    else:
        # Plot the basic route line (for reference, with low opacity)
        plt.plot(route_longs, route_lats, 'r-', linewidth=1.5, alpha=0.2)
        
        # Color-code route segments based on traffic
        for i in range(len(path) - 1):
            city1, city2 = path[i], path[i+1]
            
            traffic_factor = traffic_matrix[city1, city2]
            
            # Determine color and width based on traffic level
            if traffic_factor < 1.2:
                color = 'green'
                width = 2.0
            elif traffic_factor < 1.8:
                color = 'orange'
                width = 3.0
            else:
                color = 'red'
                width = 4.0
            
            # Draw this segment with traffic-based color and width
            plt.plot([df.iloc[city1]['Long'], df.iloc[city2]['Long']],
                     [df.iloc[city1]['Lat'], df.iloc[city2]['Lat']],
                     color=color, linewidth=width, alpha=0.8)
        
        # Add a legend for traffic levels
        from matplotlib.lines import Line2D
        traffic_legend_elements = [
            Line2D([0], [0], color='green', lw=2, label='Light Traffic (<1.2x)'),
            Line2D([0], [0], color='orange', lw=3, label='Medium Traffic (1.2-1.8x)'),
            Line2D([0], [0], color='red', lw=4, label='Heavy Traffic (>1.8x)')
        ]
        plt.legend(handles=traffic_legend_elements, loc='lower right')
    
    # Annotate cities with names
    for i, city_idx in enumerate(path):
        city_name = df.iloc[city_idx]['City']
        plt.annotate(
            city_name,
            (df.iloc[city_idx]['Long'], df.iloc[city_idx]['Lat']),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=8
        )
    
    # Highlight start/end city
    plt.scatter(
        df.iloc[START_CITY]['Long'],
        df.iloc[START_CITY]['Lat'],
        c='green',
        s=100,
        label='Start/End City'
    )
    
    # Add subtitle with distance information
    subtitle = f"Distance: {distance:.2f} km"
    if travel_time:
        subtitle += f" | Travel Time: {travel_time:.2f} hours"
    
    plt.title(f"{title}\nStarting from {city_names[START_CITY]}\n{subtitle}", fontsize=14)
    plt.xlabel('Longitude', fontsize=12)
    plt.ylabel('Latitude', fontsize=12)
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join('tsp_visualizations', filename), dpi=300)
    plt.close()

# PART 1: Train without traffic
print("\n--- Training Q-Learning WITHOUT Traffic ---")
Q_table_no_traffic, best_path_no_traffic, best_distance_no_traffic, convergence_data_no_traffic, _, training_time_no_traffic = train_q_learning(
    base_distance_matrix, 
    use_traffic=False, 
    description="Q-Learning WITHOUT Traffic"
)

# Get optimal route without traffic
print("\n--- Calculating Optimal Route WITHOUT Traffic ---")
optimal_path_no_traffic, optimal_distance_no_traffic, _, _, path_calc_time_no_traffic = get_optimal_path(
    Q_table_no_traffic, 
    base_distance_matrix, 
    use_traffic=False
)

# PART 2: Train with traffic
print("\n--- Training Q-Learning WITH Traffic ---")
Q_table_with_traffic, best_path_with_traffic, best_distance_with_traffic, convergence_data_with_traffic, traffic_history, training_time_with_traffic = train_q_learning(
    base_distance_matrix, 
    use_traffic=True, 
    description="Q-Learning WITH Traffic"
)

# Get optimal route with traffic
print("\n--- Calculating Optimal Route WITH Traffic ---")
optimal_path_with_traffic, optimal_distance_with_traffic, traffic_matrix, traffic_pattern, path_calc_time_with_traffic = get_optimal_path(
    Q_table_with_traffic, 
    base_distance_matrix, 
    use_traffic=True
)

# Calculate additional metrics
estimated_travel_time = optimal_distance_with_traffic / 60  # Assuming average speed of 60 km/h
traffic_impact_km = optimal_distance_with_traffic - optimal_distance_no_traffic
traffic_impact_percent = (optimal_distance_with_traffic / optimal_distance_no_traffic * 100 - 100)

# Print results
print("\n--- RESULTS COMPARISON ---")
print(f"Current traffic pattern: {traffic_pattern}")
print(f"DISTANCE WITHOUT traffic: {optimal_distance_no_traffic:.2f} km")
print(f"DISTANCE WITH traffic: {optimal_distance_with_traffic:.2f} km")
print(f"Traffic impact: {traffic_impact_km:.2f} km ({traffic_impact_percent:.2f}% increase)")
print(f"Estimated travel time with traffic: {estimated_travel_time:.2f} hours")
print("\nTraining performance:")
print(f"Training time WITHOUT traffic: {training_time_no_traffic:.2f} seconds")
print(f"Training time WITH traffic: {training_time_with_traffic:.2f} seconds")
print(f"Path calculation time WITHOUT traffic: {path_calc_time_no_traffic:.4f} seconds")
print(f"Path calculation time WITH traffic: {path_calc_time_with_traffic:.4f} seconds")

# Convert paths to city names for display
optimal_cities_no_traffic = [city_names[i] for i in optimal_path_no_traffic]
optimal_cities_with_traffic = [city_names[i] for i in optimal_path_with_traffic]

print("\nOptimal path WITHOUT traffic:")
print(" -> ".join(optimal_cities_no_traffic))
print("\nOptimal path WITH traffic:")
print(" -> ".join(optimal_cities_with_traffic))

# Generate the four plots
print("\n--- Generating visualizations ---")

# 1. Convergence plot WITHOUT traffic
plot_convergence(
    convergence_data_no_traffic,
    title="Convergence of Q-Learning (WITHOUT Traffic)",
    filename="convergence_no_traffic.png"
)

# 2. Convergence plot WITH traffic
plot_convergence(
    convergence_data_with_traffic,
    traffic_history=traffic_history,
    title="Convergence of Q-Learning (WITH Traffic)",
    filename="convergence_with_traffic.png"
)

# 3. Route visualization WITHOUT traffic
plot_route_map(
    optimal_path_no_traffic,
    title="TSP Route WITHOUT Traffic",
    filename="route_map_no_traffic.png",
    distance=optimal_distance_no_traffic
)

# 4. Route visualization WITH traffic
plot_route_map(
    optimal_path_with_traffic,
    traffic_matrix=traffic_matrix,
    title="TSP Route WITH Traffic",
    filename="route_map_with_traffic.png",
    distance=optimal_distance_with_traffic,
    travel_time=estimated_travel_time
)

print(f"\nAll visualizations have been saved to the 'tsp_visualizations' directory.")
print("1. convergence_no_traffic.png - Convergence plot for Q-Learning without traffic")
print("2. convergence_with_traffic.png - Convergence plot for Q-Learning with traffic")
print("3. route_map_no_traffic.png - Route visualization without traffic")
print("4. route_map_with_traffic.png - Route visualization with traffic")