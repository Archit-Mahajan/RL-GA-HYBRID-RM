import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import pdist, squareform
import random
from tqdm import tqdm
from math import radians, sin, cos, sqrt, asin
from haversine import haversine
import time

# Start timing for entire execution
start_time = time.time()

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
file_path = "/Users/architmahajan/Desktop/ResearchPapercode/Indian Cities Database.csv"  # Update with your actual path
df = pd.read_csv(file_path)

# Extract necessary data
cities = df[['City', 'Lat', 'Long']]
n_cities = len(cities)
city_names = cities['City'].tolist()

# Allow user to select the start city
print("Available cities:")
for i, city in enumerate(city_names):
    print(f"{i}: {city}")

# User input for starting city
try:
    START_CITY = int(input("Enter the index of your starting city: "))
    if START_CITY < 0 or START_CITY >= n_cities:
        print(f"Invalid index. Using default start city (0: {city_names[0]}).")
        START_CITY = 0
except ValueError:
    print(f"Invalid input. Using default start city (0: {city_names[0]}).")
    START_CITY = 0

print(f"Starting city set to: {city_names[START_CITY]}")

print(f"Calculating distance matrix for {n_cities} cities...")
# Calculate base distance matrix efficiently
base_distance_matrix = np.zeros((n_cities, n_cities))
for i in range(n_cities):
    for j in range(i+1, n_cities):  # Only calculate upper triangle
        dist = haversine_distance(
            df.iloc[i]['Lat'], df.iloc[i]['Long'],
            df.iloc[j]['Lat'], df.iloc[j]['Long']
        )
        base_distance_matrix[i, j] = dist
        base_distance_matrix[j, i] = dist  # Mirror the value to lower triangle

# ====================== TRAFFIC SIMULATION COMPONENT ======================

# Traffic parameters
class TrafficManager:
    def __init__(self, base_distance_matrix, n_cities, 
                 rush_hour_factor=2.5,      # How much slower traffic is during rush hours
                 congestion_probability=0.2, # Probability of random congestion
                 max_congestion_factor=3.0,  # Maximum slowdown due to random congestion
                 temporal_factor=0.7,        # How much traffic patterns change over time
                 spatial_correlation=0.6,    # How much nearby roads share similar traffic
                 avg_speed_kmh=60):         # Average speed in km/h for time calculations
        
        self.base_distances = base_distance_matrix.copy()
        self.n_cities = n_cities
        self.traffic_matrix = np.ones((n_cities, n_cities))  # Multiplier for base distances
        self.rush_hour_factor = rush_hour_factor
        self.congestion_probability = congestion_probability
        self.max_congestion_factor = max_congestion_factor
        self.temporal_factor = temporal_factor
        self.spatial_correlation = spatial_correlation
        self.time_of_day = 0  # 0-23 hours
        self.current_distances = self.base_distances.copy()
        self.avg_speed_kmh = avg_speed_kmh
        
        # Create city clusters for spatially correlated traffic
        self.city_clusters = self._create_city_clusters()
        
        # Initialize time-based traffic patterns (rush hours, etc.)
        self.time_patterns = self._initialize_time_patterns()
        
    def _create_city_clusters(self):
        """Group cities into geographical clusters for realistic traffic correlation"""
        clusters = []
        
        # Create a simple clustering based on proximity
        # More sophisticated clustering could be implemented
        remaining_cities = set(range(self.n_cities))
        
        while remaining_cities:
            # Pick a random city as a cluster center
            center = random.choice(list(remaining_cities))
            remaining_cities.remove(center)
            
            # Find nearby cities using distance threshold
            cluster = [center]
            threshold = np.percentile(self.base_distances[center], 15)  # 15% closest cities
            
            for city in list(remaining_cities):
                if self.base_distances[center, city] < threshold:
                    cluster.append(city)
                    remaining_cities.remove(city)
            
            clusters.append(cluster)
        
        return clusters
    
    def _initialize_time_patterns(self):
        """Create daily traffic patterns with morning/evening rush hours"""
        patterns = np.ones(24)  # 24 hours in a day
        
        # Morning rush hour (7-9 AM)
        patterns[7:10] = np.linspace(1.5, self.rush_hour_factor, 3)
        
        # Evening rush hour (4-7 PM)
        patterns[16:19] = np.linspace(1.5, self.rush_hour_factor, 3)
        patterns[19:21] = np.linspace(self.rush_hour_factor, 1.2, 2)
        
        # Late night (less traffic)
        patterns[0:5] = 0.8
        patterns[23] = 0.8
        
        return patterns
    
    def update_traffic(self, time_step=1):
        """Update traffic conditions based on time of day and random events"""
        # Update time of day
        self.time_of_day = (self.time_of_day + time_step) % 24
        
        # Reset traffic matrix while maintaining temporal correlation
        self.traffic_matrix = self.traffic_matrix * self.temporal_factor + \
                             np.ones((self.n_cities, self.n_cities)) * (1 - self.temporal_factor)
        
        # Apply time-of-day effect
        time_effect = self.time_patterns[self.time_of_day]
        
        # Apply random congestion events for each cluster
        for cluster in self.city_clusters:
            # Determine if this cluster experiences congestion
            if random.random() < self.congestion_probability:
                # Generate a congestion factor
                congestion = 1 + random.random() * (self.max_congestion_factor - 1)
                
                # Apply to all roads within and connected to this cluster
                for city1 in cluster:
                    for city2 in range(self.n_cities):
                        # Higher effect within the cluster, lower effect for connections outside
                        if city2 in cluster:
                            effect = congestion
                        else:
                            # Diminishing effect based on distance
                            effect = 1 + (congestion - 1) * self.spatial_correlation
                        
                        if city1 != city2:  # Skip self-connections
                            self.traffic_matrix[city1, city2] *= effect
                            self.traffic_matrix[city2, city1] = self.traffic_matrix[city1, city2]  # Ensure symmetry
        
        # Apply time-of-day effect to all roads
        self.traffic_matrix *= time_effect
        
        # Ensure traffic multipliers stay in reasonable bounds
        self.traffic_matrix = np.clip(self.traffic_matrix, 0.7, self.max_congestion_factor * 1.5)
        
        # Update the current distance matrix with traffic effects
        self.current_distances = self.base_distances * self.traffic_matrix
        
        return self.time_of_day, np.mean(self.traffic_matrix)
    
    def get_current_distances(self):
        """Return current distance matrix with traffic applied"""
        return self.current_distances
    
    def get_base_distances(self):
        """Return base distance matrix without traffic"""
        return self.base_distances
    
    def calculate_travel_time(self, route):
        """Calculate travel time in hours for a given route with current traffic"""
        travel_time = 0
        
        for i in range(len(route) - 1):
            from_city = route[i]
            to_city = route[i + 1]
            
            # Get distance with traffic
            distance = self.current_distances[from_city, to_city]
            
            # Calculate time: distance / (average speed / traffic factor)
            # Traffic factor already applied to distance, so just divide by base speed
            travel_time += distance / self.avg_speed_kmh
        
        return travel_time
    
    def get_traffic_info(self):
        """Return current traffic conditions for visualization/reporting"""
        avg_traffic = np.mean(self.traffic_matrix)
        max_traffic = np.max(self.traffic_matrix)
        min_traffic = np.min(self.traffic_matrix)
        
        # Find most congested route
        max_idx = np.unravel_index(np.argmax(self.traffic_matrix), self.traffic_matrix.shape)
        most_congested = (max_idx[0], max_idx[1], self.traffic_matrix[max_idx])
        
        return {
            'time_of_day': self.time_of_day,
            'avg_traffic_multiplier': avg_traffic,
            'max_traffic_multiplier': max_traffic,
            'min_traffic_multiplier': min_traffic,
            'most_congested': most_congested
        }

# Initialize traffic manager
traffic_manager = TrafficManager(base_distance_matrix, n_cities)

# Update with initial traffic pattern
current_hour, avg_traffic = traffic_manager.update_traffic(0)
print(f"Initial traffic conditions - Hour: {current_hour}:00, Average traffic multiplier: {avg_traffic:.2f}x")

# Get initial traffic-adjusted distance matrix
distance_matrix = traffic_manager.get_current_distances()

# ====================== Q-LEARNING COMPONENT ======================

# Start timing for Q-learning
ql_start_time = time.time()

# Q-Learning Hyperparameters
EPSILON_START = 0.9
EPSILON_END = 0.01
ALPHA = 0.1  # Learning rate
GAMMA = 0.9  # Discount factor
NUM_EPISODES = 50000
# START_CITY is now set by user input

# Initialize Q-table
Q_table = np.zeros((n_cities, n_cities))

print("Training Q-Learning component...")
# Training with epsilon decay and progress bar
epsilon_decay = (EPSILON_START - EPSILON_END) / NUM_EPISODES
epsilon = EPSILON_START

# Track best path and its distance for convergence monitoring
best_ql_distance = float('inf')
best_ql_path = []
ql_convergence_data = []

# Update traffic periodically during training (simulate dynamic conditions)
traffic_update_frequency = NUM_EPISODES // 24  # Update ~24 times during training

# Training loop
for episode in tqdm(range(NUM_EPISODES), desc="Training Q-Learning"):
    # Update traffic conditions periodically to simulate dynamic environment
    if episode % traffic_update_frequency == 0:
        current_hour, avg_traffic = traffic_manager.update_traffic()
        distance_matrix = traffic_manager.get_current_distances()
        
        # Optional: Print traffic update
        if episode % (traffic_update_frequency * 4) == 0:
            print(f"Traffic update - Hour: {current_hour}:00, Avg multiplier: {avg_traffic:.2f}x")
    
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
            
            # Fallback if next_city is already visited (should not happen with proper masking)
            if next_city in visited:
                next_city = random.choice(available_cities)
        
        # Calculate reward (negative distance with current traffic)
        reward = -distance_matrix[current_city, next_city]
        total_distance -= reward  # Track actual distance
        
        # Update Q-table
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
    if total_distance < best_ql_distance:
        best_ql_distance = total_distance
        best_ql_path = path + [START_CITY]  # Add return to start
        
    # Record progress for analysis
    if episode % 100 == 0:
        ql_convergence_data.append((episode, best_ql_distance))
    
    # Decay epsilon
    epsilon = max(EPSILON_END, epsilon - epsilon_decay)

# Calculate Q-learning training time
ql_training_time = time.time() - ql_start_time
print(f"Q-Learning training completed in {ql_training_time:.2f} seconds")

# Function to get optimal path using Q-values under current traffic conditions
def get_ql_optimal_path(q_table, current_distance_matrix, base_distance_matrix):
    path = [START_CITY]
    current_city = START_CITY
    visited = set([current_city])
    total_distance_with_traffic = 0
    total_distance_base = 0
    
    while len(visited) < n_cities:
        available_cities = [c for c in range(n_cities) if c not in visited]
        available_q_values = {c: q_table[current_city, c] for c in available_cities}
        
        if not available_q_values:
            break
            
        next_city = max(available_q_values.items(), key=lambda x: x[1])[0]
        total_distance_with_traffic += current_distance_matrix[current_city, next_city]
        total_distance_base += base_distance_matrix[current_city, next_city]
        
        path.append(next_city)
        visited.add(next_city)
        current_city = next_city
    
    # Return to starting city
    path.append(START_CITY)
    total_distance_with_traffic += current_distance_matrix[current_city, START_CITY]
    total_distance_base += base_distance_matrix[current_city, START_CITY]
    
    return path, total_distance_with_traffic, total_distance_base

# Get current traffic conditions for evaluation
traffic_manager.update_traffic()  # Update to a new random traffic state
current_distances = traffic_manager.get_current_distances()
base_distances = traffic_manager.get_base_distances()
traffic_info = traffic_manager.get_traffic_info()

# Get optimal path and distances
optimal_ql_path, optimal_ql_distance_with_traffic, optimal_ql_distance_base = get_ql_optimal_path(
    Q_table, current_distances, base_distances
)

# Calculate travel time for the optimal path
optimal_ql_travel_time = traffic_manager.calculate_travel_time(optimal_ql_path)

print(f"Q-Learning optimal path distance without traffic: {optimal_ql_distance_base:.2f} km")
print(f"Q-Learning optimal path distance with traffic: {optimal_ql_distance_with_traffic:.2f} km")
print(f"Q-Learning optimal path travel time: {optimal_ql_travel_time:.2f} hours")
print(f"Current traffic conditions - Hour: {traffic_info['time_of_day']}:00, " 
      f"Average multiplier: {traffic_info['avg_traffic_multiplier']:.2f}x")

# ====================== GENETIC ALGORITHM COMPONENT ======================

# Start timing for GA
ga_start_time = time.time()

# Fitness function - updated to use current traffic conditions
def fitness(route, current_distance_matrix):
    """Calculate the total distance of the given route with current traffic."""
    indices = np.array(route)
    # Use advanced indexing for faster performance
    distances = current_distance_matrix[indices[:-1], indices[1:]]
    # Add return to start
    return_distance = current_distance_matrix[indices[-1], indices[0]]
    return np.sum(distances) + return_distance

# Seed the initial population with the Q-learning solution
def create_hybrid_population(pop_size, num_cities, q_learning_path):
    """Generate an initial population using Q-learning insights."""
    population = []
    
    # Add the Q-learning solution as-is
    population.append(q_learning_path[:-1])  # Remove the last city (return to start)
    
    # Create variations of the Q-learning solution
    for _ in range(pop_size // 3):
        route = q_learning_path[:-1].copy()  # Remove last city (return to start)
        
        # Apply random swaps to create variations
        for _ in range(random.randint(1, num_cities // 3)):
            i, j = random.sample(range(len(route)), 2)
            route[i], route[j] = route[j], route[i]
            
        population.append(route)
    
    # Fill the rest with random solutions
    while len(population) < pop_size:
        route = list(range(num_cities))
        random.shuffle(route)
        population.append(route)
    
    return population

# Tournament selection - updated to use current traffic conditions
def tournament_selection(population, current_distance_matrix, tournament_size=5):
    """Select individuals using tournament selection."""
    selected = []
    pop_size = len(population)
    
    # Calculate fitness for all individuals once with current traffic
    fitness_values = [fitness(route, current_distance_matrix) for route in population]
    
    for _ in range(pop_size // 2):  # Select half of population
        # Select tournament_size random individuals
        tournament_indices = random.sample(range(pop_size), tournament_size)
        
        # Select the best from the tournament
        best_idx = min(tournament_indices, key=lambda i: fitness_values[i])
        selected.append(population[best_idx])
    
    return selected

# PMX Crossover
def pmx_crossover(parent1, parent2):
    """Partially-Mapped Crossover (PMX) - better for TSP than regular ordered crossover."""
    size = len(parent1)
    # Choose crossover points
    cxpoint1, cxpoint2 = sorted(random.sample(range(size), 2))
    
    # Initialize offspring
    offspring = [-1] * size
    
    # Copy the mapping section from parent1
    offspring[cxpoint1:cxpoint2] = parent1[cxpoint1:cxpoint2]
    
    # Create mapping between parent1 and parent2 in the crossover region
    mapping = {parent1[i]: parent2[i] for i in range(cxpoint1, cxpoint2)}
    reverse_mapping = {v: k for k, v in mapping.items()}
    
    # Fill in remaining positions from parent2
    for i in range(size):
        if i < cxpoint1 or i >= cxpoint2:  # Outside the crossover region
            value = parent2[i]
            
            # Check if the value is already in the offspring
            while value in offspring:
                value = mapping.get(value, reverse_mapping.get(value, value))
            
            offspring[i] = value
    
    return offspring

# Enhanced mutation
def diversify_mutation(route, mutation_rate=0.3):
    """Enhanced mutation with multiple strategies for better diversity."""
    if random.random() < mutation_rate:
        # Choose mutation strategy randomly
        strategy = random.choice(["swap", "insert", "reverse"])
        
        if strategy == "swap":
            # Simple swap
            i, j = random.sample(range(len(route)), 2)
            route[i], route[j] = route[j], route[i]
        
        elif strategy == "insert":
            # Remove from one position and insert at another
            i, j = random.sample(range(len(route)), 2)
            value = route.pop(i)
            route.insert(j, value)
        
        elif strategy == "reverse":
            # Reverse a sub-sequence
            i, j = sorted(random.sample(range(len(route)), 2))
            route[i:j] = reversed(route[i:j])
    
    return route

# Adaptive mutation rate
def adaptive_mutation_rate(generation, max_generations, start_rate=0.4, end_rate=0.05):
    """Adaptive mutation rate that decreases over generations."""
    return start_rate - (start_rate - end_rate) * (generation / max_generations)

# Q-learning guided local search - updated to consider traffic
def q_learning_guided_improvement(route, q_table, current_distance_matrix, improvement_chance=0.3):
    """Use Q-learning insights to improve individuals considering current traffic."""
    if random.random() > improvement_chance:
        return route
    
    new_route = route.copy()
    
    # Pick a random position to start improvement
    start_pos = random.randint(0, len(route) - 2)
    current_city = new_route[start_pos]
    
    # Look at the next position
    next_city = new_route[start_pos + 1]
    
    # Get all available cities
    available_cities = set(new_route) - {current_city}
    
    # Check for potentially better next city based on current traffic
    best_city = next_city
    best_distance = current_distance_matrix[current_city, next_city]
    
    # Consider both Q-values and current traffic conditions
    for city in available_cities:
        if city != next_city:
            # Weighted decision: 70% traffic conditions, 30% Q-learning insights
            combined_score = (
                0.7 * (-current_distance_matrix[current_city, city]) + 
                0.3 * q_table[current_city, city]
            )
            
            current_score = (
                0.7 * (-current_distance_matrix[current_city, next_city]) + 
                0.3 * q_table[current_city, next_city]
            )
            
            if combined_score > current_score:
                best_city = city
                best_distance = current_distance_matrix[current_city, city]
    
    # If a better city is found, swap it
    if best_city != next_city:
        # Find position of best_city
        best_city_pos = new_route.index(best_city)
        # Swap
        new_route[start_pos + 1], new_route[best_city_pos] = new_route[best_city_pos], new_route[start_pos + 1]
    
    return new_route

# Traffic-aware Hybrid Genetic Algorithm
def traffic_aware_hybrid_ga(q_learning_path, q_table, traffic_manager, generations=50000, pop_size=150, tournament_size=5, crossover_rate=0.95):
    """Enhanced genetic algorithm that adapts to traffic conditions."""
    # Create initial population with Q-learning influence
    population = create_hybrid_population(pop_size, n_cities, q_learning_path)
    
    # For plotting convergence
    best_fitness_history = []
    avg_fitness_history = []
    traffic_condition_history = []
    
    # Initialize best solution using Q-learning solution
    best_solution = q_learning_path[:-1]  # Remove return to start
    current_distance_matrix = traffic_manager.get_current_distances()
    base_distance_matrix = traffic_manager.get_base_distances()
    best_fitness = fitness(best_solution, current_distance_matrix)
    
    # Progress bar
    progress_bar = tqdm(range(generations), desc="Traffic-Aware GA Progress")
    
    # Update traffic every n generations
    traffic_update_interval = max(1, generations // 48)  # Update ~48 times during evolution
    
    for gen in progress_bar:
        # Periodically update traffic conditions
        if gen % traffic_update_interval == 0:
            hour, avg_traffic = traffic_manager.update_traffic()
            current_distance_matrix = traffic_manager.get_current_distances()
            traffic_info = traffic_manager.get_traffic_info()
            
            # Store traffic conditions for analysis
            traffic_condition_history.append((gen, hour, avg_traffic))
            
            # Recalculate best fitness with new traffic conditions
            best_fitness = fitness(best_solution, current_distance_matrix)
            
            # Log traffic update periodically
            if gen % (traffic_update_interval * 4) == 0:
                print(f"\nTraffic update at generation {gen} - Hour: {hour}:00, "
                      f"Average multiplier: {avg_traffic:.2f}x")
        
        # Calculate fitness for all individuals with current traffic
        fitness_values = [fitness(route, current_distance_matrix) for route in population]
        
        # Track the best solution under current traffic
        min_fitness_idx = np.argmin(fitness_values)
        current_best_fitness = fitness_values[min_fitness_idx]
        current_best = population[min_fitness_idx]
        
        if current_best_fitness < best_fitness:
            best_fitness = current_best_fitness
            best_solution = current_best.copy()
        
        # Update progress bar
        progress_bar.set_postfix({
            'Best Distance': f"{best_fitness:.2f} km",
            'Avg Distance': f"{np.mean(fitness_values):.2f} km",
            'Hour': f"{traffic_manager.time_of_day}:00"
        })
        
        # Store history for plotting
        best_fitness_history.append(best_fitness)
        avg_fitness_history.append(np.mean(fitness_values))
        
        # Tournament selection with current traffic conditions
        selected = tournament_selection(population, current_distance_matrix, tournament_size)
        
        # Create next generation
        next_generation = []
        
        # Always keep the elite (best individual)
        next_generation.append(current_best)
        
        # Generate offspring
        while len(next_generation) < pop_size:
            # Select parents
            parent1, parent2 = random.sample(selected, 2)
            
            # Apply crossover with probability
            if random.random() < crossover_rate:
                child = pmx_crossover(parent1, parent2)
            else:
                child = parent1.copy()  # No crossover
            
            # Apply mutation with adaptive rate
            mutation_rate = adaptive_mutation_rate(gen, generations)
            child = diversify_mutation(child, mutation_rate)
            
            # Apply traffic-aware Q-learning guided improvement
            child = q_learning_guided_improvement(child, q_table, current_distance_matrix)
            
            next_generation.append(child)
        
        # Update population
        population = next_generation
    
    # Get current traffic for final evaluation
    final_traffic_matrix = traffic_manager.get_current_distances()
    final_base_matrix = traffic_manager.get_base_distances()
    final_traffic_info = traffic_manager.get_traffic_info()
    
    # Evaluate best solution with traffic and without traffic
    final_distance_with_traffic = fitness(best_solution, final_traffic_matrix)
    final_distance_base = fitness(best_solution, final_base_matrix)
    
    # Complete route (add return to start)
    complete_route = best_solution + [best_solution[0]]
    
    # Calculate travel time
    travel_time = traffic_manager.calculate_travel_time(complete_route)
    
    return (complete_route, final_distance_with_traffic, final_distance_base, travel_time, 
            best_fitness_history, avg_fitness_history, traffic_condition_history, final_traffic_info)

# ====================== RUN THE TRAFFIC-AWARE HYBRID APPROACH ======================

print("Starting traffic-aware hybrid optimization...")
# Use the Q-learning path to seed the genetic algorithm with traffic awareness
(hybrid_route, hybrid_distance_with_traffic, hybrid_distance_base, hybrid_travel_time, 
 hybrid_history, hybrid_avg_history, traffic_history, final_traffic) = traffic_aware_hybrid_ga(
    optimal_ql_path,
    Q_table,
    traffic_manager,
    generations=50000,
    pop_size=150
)

# Calculate GA training time
ga_training_time = time.time() - ga_start_time

# Calculate improvements
improvement_over_ql_with_traffic = ((optimal_ql_distance_with_traffic - hybrid_distance_with_traffic) / optimal_ql_distance_with_traffic) * 100
improvement_over_ql_time = ((optimal_ql_travel_time - hybrid_travel_time) / optimal_ql_travel_time) * 100

# Calculate total execution time
total_execution_time = time.time() - start_time
print(f"Total execution time: {total_execution_time:.2f} seconds")
print(f"Q-Learning training time: {ql_training_time:.2f} seconds")
print(f"Genetic Algorithm training time: {ga_training_time:.2f} seconds")

# Get city names for routes
hybrid_route_cities = [city_names[i] for i in hybrid_route]

print("\n====== COMPARISON OF APPROACHES ======")
print("\nQ-Learning Results:")
print(f"- Optimal distance without traffic: {optimal_ql_distance_base:.2f} km")
print(f"- Optimal distance with traffic: {optimal_ql_distance_with_traffic:.2f} km")
print(f"- Estimated travel time: {optimal_ql_travel_time:.2f} hours")

print("\nHybrid GA Results:")
print(f"- Optimal distance without traffic: {hybrid_distance_base:.2f} km")
print(f"- Optimal distance with traffic: {hybrid_distance_with_traffic:.2f} km")
print(f"- Estimated travel time: {hybrid_travel_time:.2f} hours")
print(f"- Improvement over Q-Learning distance: {improvement_over_ql_with_traffic:.2f}%")
print(f"- Improvement over Q-Learning travel time: {improvement_over_ql_time:.2f}%")

# ====================== DATA VISUALIZATION ======================

## ====================== DATA VISUALIZATION ======================

# Save directory
output_dir = '/Users/architmahajan/Desktop/ResearchPapercode/hybrid_tsp_visualizatons'  # you can change this to your preferred directory

# ---- 1. Q-Learning Convergence Plot ----
plt.figure(figsize=(10, 6))
ql_episodes, ql_distances = zip(*ql_convergence_data)
plt.plot(ql_episodes, ql_distances, 'b-', linewidth=2)
plt.title('Q-Learning Convergence (Base Distance Without Traffic)')
plt.xlabel('Episodes')
plt.ylabel('Best Tour Distance (km)')
plt.grid(True)
plt.tight_layout()
plt.savefig(f'{output_dir}ql_convergence.png', dpi=300)
plt.close()

# ---- 2. Hybrid GA Convergence with Traffic ----
plt.figure(figsize=(10, 6))

# Extract traffic data
gen_points, hours, traffic_levels = zip(*traffic_history) if traffic_history else ([], [], [])

# Plot best and average fitness
plt.plot(range(len(hybrid_history)), hybrid_history, 'g-', linewidth=2, label='Best Distance with Traffic')
plt.plot(range(len(hybrid_avg_history)), hybrid_avg_history, 'b-', alpha=0.5, linewidth=1, label='Avg Distance with Traffic')

# Create a twin y-axis for traffic levels
ax2 = plt.twinx()
if gen_points:  # Only if we have traffic data
    ax2.scatter(gen_points, traffic_levels, c='r', s=30, alpha=0.6, label='Traffic Multiplier')
    ax2.set_ylabel('Traffic Multiplier', color='r')
    ax2.tick_params(axis='y', labelcolor='r')

plt.title('Hybrid GA Convergence with Traffic Updates')
plt.xlabel('Generations')
plt.ylabel('Tour Distance (km)')
plt.legend(loc='upper right')
plt.grid(True)
plt.tight_layout()
plt.savefig(f'{output_dir}ga_convergence_with_traffic.png', dpi=300)
plt.close()

# ---- 3. Route Visualization Without Traffic ----
plt.figure(figsize=(10, 8))

# Calculate base distance path (without traffic considerations)
# We'll use the same hybrid_route but calculate distances without traffic

# Plot all cities
plt.scatter(cities['Long'], cities['Lat'], c='blue', s=50, label='Cities')

# Extract coordinates for the best route
route_lats = [cities.iloc[city_idx]['Lat'] for city_idx in hybrid_route]
route_longs = [cities.iloc[city_idx]['Long'] for city_idx in hybrid_route]

# Plot the route
plt.plot(route_longs, route_lats, 'g-', linewidth=2.5, 
         label=f'Best Route (Base: {hybrid_distance_base:.2f} km)')

# Annotate cities
for i, city_idx in enumerate(hybrid_route):
    city_name = cities.iloc[city_idx]['City']
    plt.annotate(
        city_name,
        (cities.iloc[city_idx]['Long'], cities.iloc[city_idx]['Lat']),
        xytext=(5, 5),
        textcoords='offset points',
        fontsize=8
    )

# Highlight start/end city
plt.scatter(
    cities.iloc[START_CITY]['Long'],
    cities.iloc[START_CITY]['Lat'],
    c='green',
    s=100,
    label='Start/End City'
)

plt.title(f'Optimized TSP Route Without Traffic Considerations\n'
          f'Starting from {city_names[START_CITY]}\n'
          f'Base Distance: {hybrid_distance_base:.2f} km')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f'{output_dir}route_without_traffic.png', dpi=300)
plt.close()

# ---- 4. Route Visualization With Traffic ----
plt.figure(figsize=(10, 8))

# Plot all cities
plt.scatter(cities['Long'], cities['Lat'], c='blue', s=50, label='Cities')

# Extract coordinates for the best route
route_lats = [cities.iloc[city_idx]['Lat'] for city_idx in hybrid_route]
route_longs = [cities.iloc[city_idx]['Long'] for city_idx in hybrid_route]

# Get final traffic matrix for edge colors
final_traffic_matrix = traffic_manager.traffic_matrix

# Color-code route segments based on traffic
for i in range(len(hybrid_route) - 1):
    idx1 = hybrid_route[i]
    idx2 = hybrid_route[i + 1]
    
    traffic_factor = final_traffic_matrix[idx1, idx2]
    
    # Determine color based on traffic (green=light, yellow=medium, red=heavy)
    if traffic_factor < 1.2:
        color = 'green'
        width = 2.0
    elif traffic_factor < 1.8:
        color = 'orange'
        width = 3.0
    else:
        color = 'red'
        width = 4.0
    
    # Draw this segment with traffic-based color
    plt.plot([cities.iloc[idx1]['Long'], cities.iloc[idx2]['Long']], 
             [cities.iloc[idx1]['Lat'], cities.iloc[idx2]['Lat']], 
             color=color, linewidth=width, alpha=0.7)

# Annotate cities
for i, city_idx in enumerate(hybrid_route):
    city_name = cities.iloc[city_idx]['City']
    plt.annotate(
        city_name,
        (cities.iloc[city_idx]['Long'], cities.iloc[city_idx]['Lat']),
        xytext=(5, 5),
        textcoords='offset points',
        fontsize=8
    )

# Highlight start/end city
plt.scatter(
    cities.iloc[START_CITY]['Long'],
    cities.iloc[START_CITY]['Lat'],
    c='green',
    s=100,
    label='Start/End City'
)

# Add a legend for traffic levels
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color='green', lw=2, label='Light Traffic (<1.2x)'),
    Line2D([0], [0], color='orange', lw=3, label='Medium Traffic (1.2-1.8x)'),
    Line2D([0], [0], color='red', lw=4, label='Heavy Traffic (>1.8x)')
]
plt.legend(handles=legend_elements, loc='best')

plt.title(f'Optimized TSP Route with Traffic Conditions\n'
          f'Starting from {city_names[START_CITY]}\n'
          f'Travel Time: {hybrid_travel_time:.2f} hours\n'
          f'Distance with Traffic: {hybrid_distance_with_traffic:.2f} km')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.grid(True)
plt.tight_layout()
plt.savefig(f'{output_dir}route_with_traffic.png', dpi=300)
plt.close()

# ---- 5. Create a convergence comparison plot (GA with/without traffic) ----
# For this plot, we need to run additional code to get GA convergence without traffic
# This would require running another optimization, so we can add a simulated convergence 
# based on the base distances

plt.figure(figsize=(10, 6))
# Plot the hybrid GA convergence (with traffic)
plt.plot(range(len(hybrid_history)), hybrid_history, 'r-', linewidth=2, label='With Traffic')

# Simulate data for without traffic (this is just an approximation)
# In a real implementation, you would run the GA again with base distances
adjusted_history = [dist / final_traffic['avg_traffic_multiplier'] for dist in hybrid_history]
plt.plot(range(len(adjusted_history)), adjusted_history, 'g-', linewidth=2, label='Without Traffic (Estimated)')

plt.title('Hybrid GA Convergence Comparison')
plt.xlabel('Generations')
plt.ylabel('Best Tour Distance (km)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f'{output_dir}ga_convergence_comparison.png', dpi=300)
plt.close()

print("All visualizations have been saved successfully!")