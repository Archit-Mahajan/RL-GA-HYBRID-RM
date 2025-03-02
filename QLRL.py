import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import pdist, squareform
import random
from tqdm import tqdm
from math import radians, sin, cos, sqrt, asin
import time

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

# Initialize distance matrix with traffic
distance_matrix, traffic_heatmap, current_pattern = update_distance_matrix(base_distance_matrix)

# Start timing the training process
training_start_time = time.time()

# Training loop
for episode in tqdm(range(NUM_EPISODES), desc="Training Q-Learning with Traffic"):
    # Update traffic conditions periodically
    if episode % TRAFFIC_UPDATE_FREQUENCY == 0:
        distance_matrix, traffic_heatmap, current_pattern = update_distance_matrix(base_distance_matrix)
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
        reward = -distance_matrix[current_city, next_city]
        total_distance -= reward  # Track actual distance
        
        # Update Q-table with traffic-aware values
        future_value = max([Q_table[next_city, c] for c in range(n_cities) if c not in visited and c != next_city], default=0)
        Q_table[current_city, next_city] += ALPHA * (reward + GAMMA * future_value - Q_table[current_city, next_city])
        
        # Move to next city
        current_city = next_city
        visited.add(current_city)
        path.append(current_city)
    
    # Return to start city to complete the tour
    reward = -distance_matrix[current_city, START_CITY]
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
print(f"\nTraining completed in {training_time:.2f} seconds")

# Function to calculate both traffic and non-traffic distances for a given path
def calculate_path_distances(path, base_matrix, traffic_matrix):
    non_traffic_distance = 0
    traffic_distance = 0
    
    for i in range(len(path) - 1):
        city1, city2 = path[i], path[i+1]
        non_traffic_distance += base_matrix[city1, city2]
        traffic_distance += traffic_matrix[city1, city2]
    
    # Estimate travel time (assuming average speed of 60 km/h)
    travel_time = traffic_distance / 60
    
    return non_traffic_distance, traffic_distance, travel_time

# Generate an optimal path considering the current traffic
def get_optimal_path_with_traffic():
    # Start timing the optimal path calculation
    path_calc_start_time = time.time()
    
    # Update traffic one final time for the optimal route
    current_distance_matrix, current_traffic, pattern = update_distance_matrix(base_distance_matrix)
    
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
    
    # Calculate distances (with and without traffic)
    non_traffic_distance, traffic_distance, travel_time = calculate_path_distances(path, base_distance_matrix, current_distance_matrix)
    
    # End timing the optimal path calculation
    path_calc_end_time = time.time()
    path_calc_time = path_calc_end_time - path_calc_start_time
    
    return path, non_traffic_distance, traffic_distance, travel_time, current_traffic, pattern, path_calc_time

optimal_path, optimal_distance_no_traffic, optimal_distance_with_traffic, travel_time, traffic_matrix, traffic_pattern, path_calculation_time = get_optimal_path_with_traffic()
optimal_cities = [city_names[i] for i in optimal_path]

print(f"Current traffic pattern: {traffic_pattern}")
print(f"Optimal path distance WITHOUT traffic: {optimal_distance_no_traffic:.2f} km")
print(f"Optimal path distance WITH traffic: {optimal_distance_with_traffic:.2f} km")
print(f"Traffic impact: {(optimal_distance_with_traffic - optimal_distance_no_traffic):.2f} km ({(optimal_distance_with_traffic/optimal_distance_no_traffic*100-100):.2f}% increase)")
print(f"Estimated travel time: {travel_time:.2f} hours")
print(f"Time to calculate optimal path: {path_calculation_time:.4f} seconds")
print("Optimal path:", " -> ".join(optimal_cities))

# Plot results with improved visualization
fig = plt.figure(figsize=(15, 15))

# Plot 1: Convergence
ax1 = fig.add_subplot(3, 1, 1)
episodes, distances = zip(*convergence_data)
ax1.plot(episodes, distances)
ax1.set_xlabel("Episode")
ax1.set_ylabel("Best Tour Distance (km)")
ax1.set_title("Convergence of Traffic-Aware Q-Learning")
ax1.grid(True)

# Plot 2: Traffic patterns over time
ax2 = fig.add_subplot(3, 1, 2)
pattern_colors = {'peak_hours': 'red', 'normal': 'orange', 'low': 'green'}
for episode, pattern in traffic_history:
    ax2.axvline(x=episode, color=pattern_colors[pattern], alpha=0.5, linewidth=1)

ax2.set_xlabel("Episode")
ax2.set_title("Traffic Pattern Changes During Training")
ax2.set_yticks([])
# Add a legend for traffic patterns
from matplotlib.lines import Line2D
pattern_legend_elements = [Line2D([0], [0], color=color, lw=2, label=pattern) 
                        for pattern, color in pattern_colors.items()]
ax2.legend(handles=pattern_legend_elements)
ax2.grid(True)

# Plot 3: Map with route (using GA example as reference)
ax3 = fig.add_subplot(3, 1, 3)

# Plot all cities
ax3.scatter(df['Long'], df['Lat'], c='blue', s=50, label='Cities')

# Extract coordinates for the optimal route
route_lats = [df.iloc[city_idx]['Lat'] for city_idx in optimal_path]
route_longs = [df.iloc[city_idx]['Long'] for city_idx in optimal_path]

# Plot the basic route line (for reference)
ax3.plot(route_longs, route_lats, 'r-', linewidth=1.5, alpha=0.3, 
         label=f'Route: {optimal_distance_no_traffic:.2f} km (no traffic) | {optimal_distance_with_traffic:.2f} km (with traffic)')

# Color-code route segments based on traffic
for i in range(len(optimal_path) - 1):
    city1, city2 = optimal_path[i], optimal_path[i+1]
    
    traffic_factor = traffic_matrix[city1, city2]
    
    # Determine color and width based on traffic level (based on the GA example)
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
    ax3.plot([df.iloc[city1]['Long'], df.iloc[city2]['Long']],
             [df.iloc[city1]['Lat'], df.iloc[city2]['Lat']],
             color=color, linewidth=width, alpha=0.7)

# Annotate cities
for i, city_idx in enumerate(optimal_path):
    city_name = df.iloc[city_idx]['City']
    ax3.annotate(
        city_name,
        (df.iloc[city_idx]['Long'], df.iloc[city_idx]['Lat']),
        xytext=(5, 5),
        textcoords='offset points',
        fontsize=8
    )

# Highlight start/end city
ax3.scatter(
    df.iloc[START_CITY]['Long'],
    df.iloc[START_CITY]['Lat'],
    c='green',
    s=100,
    label='Start/End City'
)

# Add a legend for traffic levels
traffic_legend_elements = [
    Line2D([0], [0], color='green', lw=2, label='Light Traffic (<1.2x)'),
    Line2D([0], [0], color='orange', lw=3, label='Medium Traffic (1.2-1.8x)'),
    Line2D([0], [0], color='red', lw=4, label='Heavy Traffic (>1.8x)')
]
ax3.legend(handles=traffic_legend_elements, loc='lower right')

ax3.set_title(f'Traffic-Aware TSP Route\nStarting from {city_names[START_CITY]}\nTravel Time: {travel_time:.2f} hours')
ax3.set_xlabel('Longitude')
ax3.set_ylabel('Latitude')
ax3.grid(True)

# Add timing information to the plot
plt.figtext(0.5, 0.01, 
           f"Training time: {training_time:.2f} sec | Optimal path calculation: {path_calculation_time:.4f} sec | Traffic impact: {(optimal_distance_with_traffic/optimal_distance_no_traffic*100-100):.2f}% increase", 
           ha="center", fontsize=12, bbox={"facecolor":"white", "alpha":0.5, "pad":5})

plt.tight_layout()
plt.savefig('tsp_solution_with_traffic_comparison.png', dpi=300)
plt.show()