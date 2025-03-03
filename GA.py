import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
from haversine import haversine
from tqdm import tqdm
import time

# Start timing
start_time = time.time()

# Load the dataset
file_path = "/Users/architmahajan/Desktop/ResearchPapercode/Indian Cities Database.csv"
df = pd.read_csv(file_path)

# Extract necessary data
cities = df[['City', 'Lat', 'Long']]
num_cities = len(cities)
city_names = cities['City'].tolist()

# Allow user to select start city
def select_start_city():
    print("\nAvailable cities:")
    for i, city in enumerate(city_names):
        print(f"{i}: {city}")
    
    while True:
        try:
            start_city_idx = int(input("\nEnter the index of your desired start city: "))
            if 0 <= start_city_idx < len(city_names):
                print(f"Selected start city: {city_names[start_city_idx]}")
                return start_city_idx
            else:
                print(f"Please enter a valid index between 0 and {len(city_names)-1}")
        except ValueError:
            print("Please enter a valid integer")

# Get user-selected start city
start_city_idx = select_start_city()

# Calculate base distance matrix more efficiently
print("Calculating base distance matrix...")
base_distance_matrix = np.zeros((num_cities, num_cities))
for i in range(num_cities):
    for j in range(i+1, num_cities):  # Only calculate upper triangle
        city1 = (cities.iloc[i]['Lat'], cities.iloc[i]['Long'])
        city2 = (cities.iloc[j]['Lat'], cities.iloc[j]['Long'])
        dist = haversine(city1, city2)
        base_distance_matrix[i, j] = dist
        base_distance_matrix[j, i] = dist  # Mirror the value to lower triangle

# Traffic simulation parameters
class TrafficManager:
    def __init__(self, num_cities, base_matrix, update_interval=10):
        self.num_cities = num_cities
        self.base_matrix = base_matrix
        self.traffic_matrix = np.ones((num_cities, num_cities))  # Traffic multiplier (1.0 = normal traffic)
        self.update_interval = update_interval
        self.time_of_day = 0  # 0-23 hours
        self.current_generation = 0
        # Speed matrix in km/h
        self.speed_matrix = np.ones((num_cities, num_cities)) * 60  # Default 60 km/h between cities
        
        # Initialize with random traffic conditions
        self.initialize_traffic()
        
    def initialize_traffic(self):
        """Set up initial traffic patterns based on city connections."""
        # Create some busy routes (higher traffic multipliers)
        for i in range(self.num_cities):
            for j in range(i+1, self.num_cities):
                # Randomly initialize traffic (0.8 to 2.5 multiplier)
                # Some routes have light traffic (< 1.0), some have heavy (> 1.0)
                self.traffic_matrix[i, j] = random.uniform(0.8, 2.5)
                self.traffic_matrix[j, i] = self.traffic_matrix[i, j]  # Symmetric traffic
                
                # Initialize speeds between cities (40-100 km/h)
                base_speed = random.uniform(40, 100)
                self.speed_matrix[i, j] = base_speed
                self.speed_matrix[j, i] = base_speed
    
    def update_traffic(self):
        """Update traffic conditions periodically based on generation count."""
        self.current_generation += 1
        
        # Update traffic every update_interval generations
        if self.current_generation % self.update_interval == 0:
            self.time_of_day = (self.time_of_day + 1) % 24  # Advance time of day
            
            # Apply time-based traffic patterns
            time_factor = self.get_time_factor(self.time_of_day)
            
            # Update each route's traffic
            for i in range(self.num_cities):
                for j in range(i+1, self.num_cities):
                    if random.random() < 0.3:  # 30% chance of traffic change
                        # Traffic fluctuation with time factor influence
                        change = random.uniform(-0.3, 0.3) * time_factor
                        current = self.traffic_matrix[i, j]
                        # Ensure traffic stays within reasonable bounds
                        new_traffic = max(0.7, min(3.0, current + change))
                        self.traffic_matrix[i, j] = new_traffic
                        self.traffic_matrix[j, i] = new_traffic  # Keep symmetric
    
    def get_time_factor(self, hour):
        """Return traffic factor based on time of day (rush hours have higher values)."""
        if 7 <= hour <= 9:  # Morning rush hour
            return 1.5
        elif 16 <= hour <= 19:  # Evening rush hour
            return 1.7
        elif 22 <= hour <= 5:  # Night (light traffic)
            return 0.7
        else:  # Normal daytime
            return 1.0
    
    def get_current_distance_matrix(self):
        """Return distance matrix adjusted for current traffic conditions."""
        return self.base_matrix * self.traffic_matrix
    
    def get_travel_time_matrix(self):
        """Return travel time matrix in hours based on distance and speed with traffic."""
        # Adjust speeds based on traffic (higher traffic = lower speed)
        effective_speed = self.speed_matrix / self.traffic_matrix
        # Calculate time (distance/speed) in hours
        time_matrix = self.base_matrix / effective_speed
        return time_matrix
    
    def get_traffic_status_report(self):
        """Generate a report of current traffic conditions."""
        avg_traffic = np.mean(self.traffic_matrix)
        max_traffic = np.max(self.traffic_matrix)
        min_traffic = np.min(self.traffic_matrix)
        
        # Find busiest route
        max_idx = np.unravel_index(np.argmax(self.traffic_matrix), self.traffic_matrix.shape)
        
        return {
            'time': self.time_of_day,
            'avg_factor': avg_traffic,
            'max_factor': max_traffic,
            'min_factor': min_traffic,
            'busiest_route': (max_idx[0], max_idx[1]),
            'busiest_factor': self.traffic_matrix[max_idx]
        }

# Initialize traffic manager
traffic_manager = TrafficManager(num_cities, base_distance_matrix)


# Fitness functions for both with and without traffic
def fitness_with_traffic(route):
    """Calculate the total distance of the given route with current traffic conditions."""
    # Get current traffic-adjusted distance matrix
    current_matrix = traffic_manager.get_current_distance_matrix()
    
    indices = np.array(route)
    # Use advanced indexing for faster performance
    distances = current_matrix[indices[:-1], indices[1:]]
    # Add return to start
    return_distance = current_matrix[indices[-1], indices[0]]
    return np.sum(distances) + return_distance

def fitness_without_traffic(route):
    """Calculate the total distance of the given route without traffic conditions."""
    indices = np.array(route)
    # Use advanced indexing for faster performance
    distances = base_distance_matrix[indices[:-1], indices[1:]]
    # Add return to start
    return_distance = base_distance_matrix[indices[-1], indices[0]]
    return np.sum(distances) + return_distance

def create_population(pop_size, num_cities, start_city_idx):
    """Generate an initial population of random routes with fixed start city."""
    population = []
    for _ in range(pop_size):
        # Create a list of cities excluding the start city
        other_cities = [i for i in range(num_cities) if i != start_city_idx]
        # Shuffle the other cities
        random.shuffle(other_cities)
        # Create route with start city at the beginning
        route = [start_city_idx] + other_cities
        population.append(route)
    return population

def tournament_selection(population, fitness_func, tournament_size=5):
    """Select individuals using tournament selection."""
    selected = []
    pop_size = len(population)
    
    # Calculate fitness for all individuals once
    fitness_values = [fitness_func(route) for route in population]
    
    for _ in range(pop_size // 2):  # Select half of population
        # Select tournament_size random individuals
        tournament_indices = random.sample(range(pop_size), tournament_size)
        
        # Select the best from the tournament
        best_idx = min(tournament_indices, key=lambda i: fitness_values[i])
        selected.append(population[best_idx])
    
    return selected

def pmx_crossover(parent1, parent2):
    """Partially-Mapped Crossover (PMX) with fixed start city."""
    # Ensure the first city (start city) stays in place
    start_city = parent1[0]
    if parent2[0] != start_city:
        raise ValueError("Both parents should have the same start city")
    
    size = len(parent1)
    
    # Choose crossover points (ensuring start city is not affected)
    cxpoint1 = random.randint(1, size - 2)  # Start from index 1
    cxpoint2 = random.randint(cxpoint1 + 1, size - 1)
    
    # Initialize offspring with same start city
    offspring = [start_city] + [-1] * (size - 1)
    
    # Copy the mapping section from parent1
    offspring[cxpoint1:cxpoint2] = parent1[cxpoint1:cxpoint2]
    
    # Create mapping between parent1 and parent2 in the crossover region
    mapping = {parent1[i]: parent2[i] for i in range(cxpoint1, cxpoint2)}
    reverse_mapping = {v: k for k, v in mapping.items()}
    
    # Fill in remaining positions from parent2
    for i in range(1, size):  # Start from 1 to skip start city
        if i < cxpoint1 or i >= cxpoint2:  # Outside the crossover region
            value = parent2[i]
            
            # Check if the value is already in the offspring
            while value in offspring:
                value = mapping.get(value, reverse_mapping.get(value, value))
            
            offspring[i] = value
    
    return offspring

def diversify_mutation(route, mutation_rate=0.3):
    """Enhanced mutation with multiple strategies for better diversity, keeping start city fixed."""
    if random.random() < mutation_rate:
        # Choose mutation strategy randomly, protecting start city
        strategy = random.choice(["swap", "insert", "reverse"])
        
        if strategy == "swap":
            # Simple swap (excluding start city)
            i, j = random.sample(range(1, len(route)), 2)
            route[i], route[j] = route[j], route[i]
        
        elif strategy == "insert":
            # Remove from one position and insert at another (excluding start city)
            i, j = random.sample(range(1, len(route)), 2)
            value = route.pop(i)
            route.insert(j, value)
        
        elif strategy == "reverse":
            # Reverse a sub-sequence (excluding start city)
            i = random.randint(1, len(route) - 3)
            j = random.randint(i + 1, len(route) - 1)
            route[i:j] = reversed(route[i:j])
    
    return route

def adaptive_mutation_rate(generation, max_generations, start_rate=0.4, end_rate=0.05):
    """Adaptive mutation rate that decreases over generations."""
    return start_rate - (start_rate - end_rate) * (generation / max_generations)

def genetic_algorithm(generations=10000, pop_size=150, tournament_size=5, crossover_rate=0.95, 
                      start_city_idx=0, with_traffic=True):
    """Enhanced genetic algorithm with fixed start city, progress tracking, adaptive parameters."""
    # Create initial population with fixed start city
    population = create_population(pop_size, num_cities, start_city_idx)
    
    # Select fitness function based on whether we use traffic or not
    fitness_func = fitness_with_traffic if with_traffic else fitness_without_traffic
    
    # For plotting convergence
    best_fitness_history = []
    avg_fitness_history = []
    traffic_factor_history = []
    
    # Initialize best solution
    best_solution = None
    best_fitness = float('inf')
    
    # Progress bar description
    desc = "GA Progress (with traffic)" if with_traffic else "GA Progress (without traffic)"
    
    # Progress bar
    progress_bar = tqdm(range(generations), desc=desc)
    
    for gen in progress_bar:
        # Update traffic conditions based on generation (only if using traffic)
        if with_traffic:
            traffic_manager.update_traffic()
            traffic_status = traffic_manager.get_traffic_status_report()
            traffic_factor_history.append(traffic_status['avg_factor'])
        
        # Calculate fitness for all individuals with current traffic
        fitness_values = [fitness_func(route) for route in population]
        
        # Track the best solution
        min_fitness_idx = np.argmin(fitness_values)
        current_best_fitness = fitness_values[min_fitness_idx]
        current_best = population[min_fitness_idx]
        
        if current_best_fitness < best_fitness:
            best_fitness = current_best_fitness
            best_solution = current_best.copy()
        
        # Update progress bar
        postfix = {'Best Dist': f"{best_fitness:.2f} km", 'Avg Dist': f"{np.mean(fitness_values):.2f} km"}
        if with_traffic:
            postfix.update({
                'Traffic': f"{traffic_status['avg_factor']:.2f}x",
                'Time': f"{traffic_status['time']:02d}:00"
            })
        progress_bar.set_postfix(postfix)
        
        # Store history for plotting
        best_fitness_history.append(best_fitness)
        avg_fitness_history.append(np.mean(fitness_values))
        
        # Tournament selection
        selected = tournament_selection(population, fitness_func, tournament_size)
        
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
            
            next_generation.append(child)
        
        # Update population
        population = next_generation
    
    # Calculate final route metrics
    if with_traffic:
        final_distance_matrix = traffic_manager.get_current_distance_matrix()
        final_time_matrix = traffic_manager.get_travel_time_matrix()
    else:
        final_distance_matrix = base_distance_matrix
        final_time_matrix = None  # Not using travel time for non-traffic version
    
    final_distance = calculate_route_distance(best_solution, final_distance_matrix)
    
    # Calculate travel time only if using traffic
    final_travel_time = None
    if with_traffic:
        final_travel_time = calculate_route_time(best_solution, final_time_matrix)
    
    # Return results
    return best_solution, final_distance, final_travel_time, best_fitness_history, avg_fitness_history, traffic_factor_history

def calculate_route_distance(route, distance_matrix):
    """Utility function to calculate route distance with given distance matrix."""
    total = 0
    for i in range(len(route) - 1):
        total += distance_matrix[route[i], route[i+1]]
    # Add return to start
    total += distance_matrix[route[-1], route[0]]
    return total

def calculate_route_time(route, time_matrix):
    """Utility function to calculate route travel time with given time matrix."""
    total_time = 0
    for i in range(len(route) - 1):
        total_time += time_matrix[route[i], route[i+1]]
    # Add return to start
    total_time += time_matrix[route[-1], route[0]]
    return total_time

# Function to plot route visualization
def plot_route_visualization(route, with_traffic=True, save_filename=None):
    """Create visualization of the route on a map."""
    plt.figure(figsize=(12, 10))
    
    # Extract coordinates for the route
    route_lats = [cities.iloc[city_idx]['Lat'] for city_idx in route]
    route_longs = [cities.iloc[city_idx]['Long'] for city_idx in route]
    
    # Add start city at the end to complete the loop
    route_lats.append(route_lats[0])
    route_longs.append(route_longs[0])
    
    # Plot all cities
    plt.scatter(cities['Long'], cities['Lat'], c='blue', s=50, label='Cities')
    
    if with_traffic:
        # Calculate distance with traffic
        final_distance = calculate_route_distance(route, traffic_manager.get_current_distance_matrix())
        final_traffic = traffic_manager.traffic_matrix
        title_suffix = f"with Traffic\nStarting from {city_names[route[0]]}"
        
        # Calculate travel time
        travel_time = calculate_route_time(route, traffic_manager.get_travel_time_matrix())
        title_suffix += f"\nTravel Time: {travel_time:.2f} hours"
        
        # Color-code route segments based on traffic
        for i in range(len(route)):
            idx1 = route[i]
            idx2 = route[(i + 1) % len(route)]
            
            traffic_factor = final_traffic[idx1, idx2]
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
            
        # Add a legend for traffic levels
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='green', lw=2, label='Light Traffic (<1.2x)'),
            Line2D([0], [0], color='orange', lw=3, label='Medium Traffic (1.2-1.8x)'),
            Line2D([0], [0], color='red', lw=4, label='Heavy Traffic (>1.8x)')
        ]
        plt.legend(handles=legend_elements, loc='lower right')
    else:
        # Calculate distance without traffic
        final_distance = calculate_route_distance(route, base_distance_matrix)
        # Plot the route with a single color
        plt.plot(route_longs, route_lats, 'r-', linewidth=2, label=f'Best Route ({final_distance:.2f} km)')
        title_suffix = f"without Traffic\nStarting from {city_names[route[0]]}"
        plt.legend()
    
    # Annotate cities
    for i, city_idx in enumerate(route):
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
        cities.iloc[route[0]]['Long'],
        cities.iloc[route[0]]['Lat'],
        c='green',
        s=100,
        label='Start/End City'
    )
    
    plt.title(f'Optimized TSP Route {title_suffix}\nTotal Distance: {final_distance:.2f} km')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.grid(True)
    
    if save_filename:
        plt.savefig(save_filename, dpi=300)
    
    plt.tight_layout()

# Function to plot convergence data
def plot_convergence(best_history, avg_history, traffic_history=None, with_traffic=True, save_filename=None):
    """Create convergence plot showing optimization progress."""
    plt.figure(figsize=(12, 8))
    
    plt.plot(best_history, 'r-', label='Best Distance')
    plt.plot(avg_history, 'b-', label='Average Distance')
    
    if with_traffic and traffic_history:
        # Add a secondary y-axis for traffic factor
        ax1 = plt.gca()
        ax2 = ax1.twinx()
        ax2.plot(traffic_history, 'g-', label='Traffic Factor')
        ax2.set_ylabel('Traffic Multiplier', color='g')
        ax2.tick_params(axis='y', labelcolor='g')
        ax2.axhline(y=1.0, color='g', linestyle='--', alpha=0.5, label='Normal Traffic')
        
        # Add combined legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        title = 'Genetic Algorithm Convergence with Traffic Conditions'
    else:
        plt.legend()
        title = 'Genetic Algorithm Convergence without Traffic'
    
    plt.title(title)
    plt.xlabel('Generation')
    plt.ylabel('Distance (km)')
    plt.grid(True)
    
    if save_filename:
        plt.savefig(save_filename, dpi=300)
    
    plt.tight_layout()

# Run the genetic algorithm twice - once with traffic and once without
print("Starting genetic algorithm optimization with traffic conditions...")
print(f"Using {city_names[start_city_idx]} as the starting city")

# Run with traffic
best_route_traffic, best_distance_traffic, travel_time, best_history_traffic, avg_history_traffic, traffic_history = genetic_algorithm(
    generations=20000,  # Reduced for demonstration
    pop_size=150,
    start_city_idx=start_city_idx,
    with_traffic=True
)

print("\nStarting genetic algorithm optimization without traffic conditions...")
print(f"Using {city_names[start_city_idx]} as the starting city")

# Run without traffic
best_route_no_traffic, best_distance_no_traffic, _, best_history_no_traffic, avg_history_no_traffic, _ = genetic_algorithm(
    generations=20000,  # Reduced for demonstration
    pop_size=150,
    start_city_idx=start_city_idx,
    with_traffic=False
)

# Calculate total execution time
execution_time = time.time() - start_time
print(f"Total execution time: {execution_time:.2f} seconds")

# Generate and save the four separate visualizations
print("\nGenerating visualizations...")

# 1. Route visualization with traffic
plot_route_visualization(
    best_route_traffic, 
    with_traffic=True, 
    save_filename='tsp_route_with_traffic.png'
)

# 2. Route visualization without traffic
plot_route_visualization(
    best_route_no_traffic, 
    with_traffic=False, 
    save_filename='tsp_route_without_traffic.png'
)

# 3. Convergence graph with traffic
plot_convergence(
    best_history_traffic, 
    avg_history_traffic, 
    traffic_history, 
    with_traffic=True, 
    save_filename='tsp_convergence_with_traffic.png'
)

# 4. Convergence graph without traffic
plot_convergence(
    best_history_no_traffic, 
    avg_history_no_traffic, 
    with_traffic=False, 
    save_filename='tsp_convergence_without_traffic.png'
)

print("\nAll visualizations have been saved to:")
print("- tsp_route_with_traffic.png")
print("- tsp_route_without_traffic.png")
print("- tsp_convergence_with_traffic.png")
print("- tsp_convergence_without_traffic.png")

# Print summary results
print("\nResults Summary:")
print(f"With Traffic: Total distance = {best_distance_traffic:.2f} km, Travel time = {travel_time:.2f} hours")
print(f"Without Traffic: Total distance = {best_distance_no_traffic:.2f} km")
print(f"Traffic impact: {(best_distance_traffic/best_distance_no_traffic - 1)*100:.2f}% increase in distance")

# Display route information
print("\nBest route with traffic:")
best_route_cities_traffic = [city_names[i] for i in best_route_traffic]
print(" -> ".join(best_route_cities_traffic) + f" -> {best_route_cities_traffic[0]}")

print("\nBest route without traffic:")
best_route_cities_no_traffic = [city_names[i] for i in best_route_no_traffic]
print(" -> ".join(best_route_cities_no_traffic) + f" -> {best_route_cities_no_traffic[0]}")