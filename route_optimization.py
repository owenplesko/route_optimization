import numpy as np
import random
import math
import json
from tqdm import tqdm

world_map = np.load("world_arr.npy")
weight_map = np.load("weight_map.npy")

def getLocations(chunkX1, chunkZ1, chunkX2, chunkZ2, threshold=45):
    # dont get any wp locations on border of 8 x 8 chunk area as some gemstones will not be loaded
    x1 = (chunkX1 * 16) + 1
    z1 = (chunkZ1 * 16) + 1
    x2 = (chunkX2 * 16)
    z2 = (chunkZ2 * 16)
    
    route_area = weight_map[x1:x2, z1:z2, :]
    
    potential_location_indices = np.argwhere(route_area > threshold)
    potential_locations = np.array([(potential_location_indices[i][0] + x1,
                                     potential_location_indices[i][1] + z1,
                                     potential_location_indices[i][2],
                                     route_area[potential_location_indices[i][0],
                                                   potential_location_indices[i][1],
                                                   potential_location_indices[i][2]])
                                    for i in range(len(potential_location_indices))])
    
    # get best locations without doublecounting weight
    locations = []
    world_copy = np.copy(world_map)
    while len(potential_locations) > 0 and potential_locations[0][3] >= threshold:
        # get highest weight location
        best_location = potential_locations[0][:3]
        expected_weight = potential_locations[0][3]
        
        actual_weight = np.sum(world_copy[best_location[0] - 1:best_location[0] + 2, best_location[1] - 1:best_location[1] + 2, best_location[2] + 1:best_location[2] + 6])
        actual_weight -= np.sum(world_copy[best_location[0], best_location[1], best_location[2] + 1:best_location[2] + 3])
        
        # if actual < expected update expected and resort array
        if actual_weight < expected_weight:
            potential_locations[0][3] = actual_weight
            potential_locations = potential_locations[potential_locations[:, -1].argsort()[::-1]]
            continue
        
        # else add location to locations
        locations.append(potential_locations[0])
        potential_locations = potential_locations[1:]
        
        # remove gemstones used by location in world copy
        x, z, y = best_location
        world_copy[x, z, y] = 0
        for xShift in range(-1, 2):
            for yShift in range(1, 5):
                for zShift in range(-1, 2):
                    world_copy[x + xShift, z + zShift, y + yShift] = 0
    return locations

def randomRoute(numLocations):
    route = []
    for i in range(numLocations):
        route.append(i)
    for i in range(numLocations):
        randomIndex = random.randint(0, numLocations - 1)
        temp = route[i]
        route[i] = route[randomIndex]
        route[randomIndex] = temp
    return route

def distance(loc1, loc2):
    return math.sqrt((loc1[0] - loc2[0])**2 + (loc1[2] - loc2[2])**2 + (loc1[1] - loc2[1])**2)

def rateRoute(locations, route, routeLength):
    total_distance = 0
    for i in range(1, routeLength):
        total_distance += distance(locations[route[i-1]], locations[route[i]])
    total_distance += distance(locations[route[0]], locations[route[routeLength - 1]])
    return total_distance

def acceptance_probability(old_energy, new_energy, temperature):
    delta = old_energy - new_energy
    if delta > 0:
        return 1.0
    exponent = delta / temperature
    capped_exponent = min(max(exponent, -1000), 1000)  # Cap the exponent cause i was getting overflows :(
    p = math.exp(capped_exponent)
    return p

def simulated_annealing(locations, route_length, num_iterations=1000000, initial_temperature=10.0, cooling_rate=0.0000025):
    current_route = randomRoute(len(locations))
    best_route = list(current_route)
    current_energy = rateRoute(locations, current_route, route_length)
    best_energy = current_energy

    for iteration in tqdm(range(num_iterations)):
        temperature = initial_temperature * math.exp(-cooling_rate * iteration)

        # Create a neighboring solution by swapping two random locations in the route
        index1 = random.randint(0, route_length - 1)
        index2 = random.randint(0, len(locations) - 1)
        while index2 == index1:
            index2 = random.randint(0, len(locations) - 1)
        current_route[index1], current_route[index2] = current_route[index2], current_route[index1]

        new_energy = rateRoute(locations, current_route, route_length)
        if acceptance_probability(current_energy, new_energy, temperature) > random.random():
            current_energy = new_energy
        else:
            current_route[index1], current_route[index2] = current_route[index2], current_route[index1]

        if current_energy < best_energy:
            print(f"new best: {current_energy}")
            best_route = list(current_route)
            best_energy = current_energy

    return best_route

def formatRoute(locations, route, route_length):
    json_data = [{"x": int(locations[route[i]][0] + 192), "y": int(locations[route[i]][2] + 30), "z": int(locations[route[i]][1] + 192), "options": {"name": i+1}} for i in range(route_length)]
    return json_data

if __name__ == "__main__":
    locations = getLocations(0, 0, 8, 8, threshold=40)
    route = simulated_annealing(locations, route_length=150)
    json_route = formatRoute(locations, route, 150)
    with open("route.txt", 'w') as f:
        json.dump(json_route, f)
