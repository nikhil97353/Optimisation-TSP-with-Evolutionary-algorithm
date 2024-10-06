#!/usr/bin/env python
# coding: utf-8

# # Import the necessary Libraries 

# In[19]:


import xml.etree.ElementTree as ET
import networkx as nx
import matplotlib.pyplot as plt
import random
import numpy as np


# # Reading Contents of the XML file

# In[20]:


import xml.etree.ElementTree as ET

def read_xml_and_display(xml_file_path):
    # Parse the XML file
    tree = ET.parse(xml_file_path)
    root = tree.getroot()

    # Display XML content in the console
    xml_content = ET.tostring(root, encoding='utf-8').decode('utf-8')
    print(xml_content)

# Replace 'your_xml_file.xml' with the path to your XML file
xml_file_path = 'burma14.xml'
read_xml_and_display(xml_file_path)


# # Reading Contents of the XML file

# Network Graph for burma14

# In[21]:


def parse_xml(xml_file_path):
    tree = ET.parse(xml_file_path)
    root = tree.getroot()

    # Extract graph information
    graph_elem = root.find(".//graph")
    vertices = graph_elem.findall(".//vertex")

    G = nx.Graph()

    for vertex_elem in vertices:
        edges = vertex_elem.findall(".//edge")
        vertex_id = int(edges[0].text)  # Assuming the vertex ID is the first edge's text
        G.add_node(vertex_id)

        for edge_elem in edges[1:]:  # Skip the first edge, which is the vertex ID
            target_vertex_id = int(edge_elem.text)
            cost = float(edge_elem.get("cost"))
            G.add_edge(vertex_id, target_vertex_id, weight=cost)

    return G

def visualize_graph(graph):
    pos = nx.spring_layout(graph)  # You can use other layout algorithms based on your preference
    labels = nx.get_edge_attributes(graph, 'weight')

    nx.draw(graph, pos, with_labels=True)
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=labels)

    plt.title("Graph Visualization")
    plt.show()

# Replace 'your_xml_file.xml' with the path to your XML file
xml_file_path = 'burma14.xml'
graph = parse_xml(xml_file_path)
visualize_graph(graph)


# Network Graph for brazil58

# In[22]:


def parse_xml(xml_file_path):
    tree = ET.parse(xml_file_path)
    root = tree.getroot()

    # Extract graph information
    graph_elem = root.find(".//graph")
    vertices = graph_elem.findall(".//vertex")

    G = nx.Graph()

    for vertex_elem in vertices:
        edges = vertex_elem.findall(".//edge")
        vertex_id = int(edges[0].text)  # Assuming the vertex ID is the first edge's text
        G.add_node(vertex_id)

        for edge_elem in edges[1:]:  # Skip the first edge, which is the vertex ID
            target_vertex_id = int(edge_elem.text)
            cost = float(edge_elem.get("cost"))
            G.add_edge(vertex_id, target_vertex_id, weight=cost)

    return G

def visualize_graph(graph):
    pos = nx.spring_layout(graph, seed=42)  # Fixed seed for reproducibility
    labels = nx.get_edge_attributes(graph, 'weight')

    # Increase the size of the figure for better visibility
    plt.figure(figsize=(12, 10))

    # Adjust the node and edge colors for better contrast
    nx.draw_networkx_nodes(graph, pos, node_color='skyblue', node_size=1200)
    nx.draw_networkx_edges(graph, pos, edge_color='gray', width=2)
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=labels)

    # Increase font size for better readability
    nx.draw_networkx_labels(graph, pos, font_size=12)

    plt.title("Graph Visualization")
    plt.axis('off')  # Turn off axis labels
    plt.show()

# Replace 'your_xml_file.xml' with the path to your XML file
xml_file_path = 'brazil58.xml'
graph = parse_xml(xml_file_path)
visualize_graph(graph)


# # brazil58

# Defining the required function for the implementation of EA and plotting the convergence curve for various parameters

# In[23]:


def parse_xml(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    graph = root.find('graph')
    num_cities = len(graph.findall('vertex'))
    distance_matrix = [[0 for _ in range(num_cities)] for _ in range(num_cities)]

    for i, vertex in enumerate(graph.findall('vertex')):
        for edge in vertex.findall('edge'):
            j = int(edge.text)
            cost = float(edge.get('cost'))
            distance_matrix[i][j] = cost

    return distance_matrix

def calculate_fitness(route, distance_matrix):
    return sum(distance_matrix[route[i]][route[(i+1) % len(route)]] for i in range(len(route)))

# Improved Initialization Function
def initialize_population(pop_size, num_cities):
    return [random.sample(range(num_cities), num_cities) for _ in range(pop_size)]

# Improved Tournament Selection
def tournament_selection(population, tournament_size, distance_matrix):
    tournament = random.sample(population, tournament_size)
    fittest = min(tournament, key=lambda route: calculate_fitness(route, distance_matrix))
    return fittest

# Improved Crossover (Order Crossover)
def crossover(parent1, parent2):
    child = [None]*len(parent1)
    start, end = sorted(random.sample(range(len(parent1)), 2))
    child[start:end] = parent1[start:end]

    parent2_filtered = [item for item in parent2 if item not in child]
    child = [parent2_filtered.pop(0) if gene is None else gene for gene in child]
    return child

# Improved Mutation (Swap Mutation)
def mutate(route, mutation_rate):
    if random.random() < mutation_rate:
        i, j = random.sample(range(len(route)), 2)
        route[i], route[j] = route[j], route[i]
    return route

# Improved EA Function with Elitism and Fitness Tracking
def run_ea_and_plot(distance_matrix, population_size, tournament_size, num_iterations, mutation_rate):
    num_cities = len(distance_matrix)
    population = initialize_population(population_size, num_cities)
    best_route = min(population, key=lambda route: calculate_fitness(route, distance_matrix))
    best_fitness = calculate_fitness(best_route, distance_matrix)
    fitness_over_time = [best_fitness]

    for _ in range(num_iterations):
        new_population = [best_route]  # Elitism: Keep the best solution
        while len(new_population) < population_size:
            parent1 = tournament_selection(population, tournament_size, distance_matrix)
            parent2 = tournament_selection(population, tournament_size, distance_matrix)
            child = crossover(parent1, parent2)
            child = mutate(child, mutation_rate)
            new_population.append(child)

        population = new_population
        current_best = min(population, key=lambda route: calculate_fitness(route, distance_matrix))
        current_fitness = calculate_fitness(current_best, distance_matrix)
        if current_fitness < best_fitness:
            best_fitness = current_fitness
            best_route = current_best

        fitness_over_time.append(best_fitness)

    plt.plot(fitness_over_time)
    plt.title('Convergence Curve')
    plt.xlabel('Iteration')
    plt.ylabel('Best Fitness (Total Travel Cost)')
    plt.show()

    return best_route, best_fitness


# Run the below command to experiment solution for various combination of population sizes, tournament sizes with fixed mutation rate

# In[ ]:


xml_file = 'brazil58.xml'
distance_matrix = parse_xml(xml_file)
population_sizes = [50, 100, 200]
tournament_sizes = [5, 10, 20]
num_iterations = 10000
mutation_rate = 0.2  # Lower mutation rate for more stability
for population_size in population_sizes:
        for tournament_size in tournament_sizes:
            best_route, best_fitness = run_ea_and_plot(distance_matrix, population_size, tournament_size, num_iterations, mutation_rate)

            print(f"Population Size: {population_size}, Tournament Size: {tournament_size}")
            print("Best Route:", best_route)
            print("Best Fitness (Total Travel Cost):", best_fitness)
            print()


# # burma14

# Defining the required function for the implementation of EA and plotting the convergence curve for various parameters

# In[13]:


import xml.etree.ElementTree as ET
import random
import numpy as np
import matplotlib.pyplot as plt

def parse_xml(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    graph = root.find('graph')
    num_cities = len(graph.findall('vertex'))
    distance_matrix = [[0 for _ in range(num_cities)] for _ in range(num_cities)]

    for i, vertex in enumerate(graph.findall('vertex')):
        for edge in vertex.findall('edge'):
            j = int(edge.text)
            cost = float(edge.get('cost'))
            distance_matrix[i][j] = cost

    return distance_matrix

def calculate_fitness(route, distance_matrix):
    return sum(distance_matrix[route[i]][route[(i+1) % len(route)]] for i in range(len(route)))

# Improved Initialization Function
def initialize_population(pop_size, num_cities):
    return [random.sample(range(num_cities), num_cities) for _ in range(pop_size)]

# Improved Tournament Selection
def tournament_selection(population, tournament_size, distance_matrix):
    tournament = random.sample(population, tournament_size)
    fittest = min(tournament, key=lambda route: calculate_fitness(route, distance_matrix))
    return fittest

# Improved Crossover (Order Crossover)
def crossover(parent1, parent2):
    child = [None]*len(parent1)
    start, end = sorted(random.sample(range(len(parent1)), 2))
    child[start:end] = parent1[start:end]

    parent2_filtered = [item for item in parent2 if item not in child]
    child = [parent2_filtered.pop(0) if gene is None else gene for gene in child]
    return child

# Improved Mutation (Swap Mutation)
def mutate(route, mutation_rate):
    if random.random() < mutation_rate:
        i, j = random.sample(range(len(route)), 2)
        route[i], route[j] = route[j], route[i]
    return route

# Improved EA Function with Elitism and Fitness Tracking
def run_ea_and_plot(distance_matrix, population_size, tournament_size, num_iterations, mutation_rate):
    num_cities = len(distance_matrix)
    population = initialize_population(population_size, num_cities)
    best_route = min(population, key=lambda route: calculate_fitness(route, distance_matrix))
    best_fitness = calculate_fitness(best_route, distance_matrix)
    fitness_over_time = [best_fitness]

    for _ in range(num_iterations):
        new_population = [best_route]  # Elitism: Keep the best solution
        while len(new_population) < population_size:
            parent1 = tournament_selection(population, tournament_size, distance_matrix)
            parent2 = tournament_selection(population, tournament_size, distance_matrix)
            child = crossover(parent1, parent2)
            child = mutate(child, mutation_rate)
            new_population.append(child)

        population = new_population
        current_best = min(population, key=lambda route: calculate_fitness(route, distance_matrix))
        current_fitness = calculate_fitness(current_best, distance_matrix)
        if current_fitness < best_fitness:
            best_fitness = current_fitness
            best_route = current_best

        fitness_over_time.append(best_fitness)

    plt.plot(fitness_over_time)
    plt.title('Convergence Curve')
    plt.xlabel('Iteration')
    plt.ylabel('Best Fitness (Total Travel Cost)')
    plt.show()

    return best_route, best_fitness

# Plot the scatter plot
def plot_scatter(population_sizes, results):
    sizes, fitness_values = zip(*results)
    plt.scatter(sizes, fitness_values, marker='o', color='blue')
    plt.title('Scatter Plot: Population Size vs Best Fitness')
    plt.xlabel('Population Size')
    plt.ylabel('Best Fitness (Total Travel Cost)')
    plt.show()


# In[ ]:


xml_file = 'burma14.xml'
distance_matrix = parse_xml(xml_file)
population_sizes = [50, 100, 200]
tournament_sizes = [5, 10, 20]
num_iterations = 10000
mutation_rate = 0.2  # Lower mutation rate for more stability
for population_size in population_sizes:
        for tournament_size in tournament_sizes:
            best_route, best_fitness = run_ea_and_plot(distance_matrix, population_size, tournament_size, num_iterations, mutation_rate)

            print(f"Population Size: {population_size}, Tournament Size: {tournament_size}")
            print("Best Route:", best_route)
            print("Best Fitness (Total Travel Cost):", best_fitness)
            print()

