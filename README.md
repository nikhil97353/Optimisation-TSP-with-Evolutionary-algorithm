# Evolutionary Algorithm for TSP Optimization

This project implements an **Evolutionary Algorithm (EA)** to solve the **Traveling Salesman Problem (TSP)** using various evolutionary strategies like crossover, mutation, and selection. The algorithm is capable of exploring different combinations of population size, tournament size, and mutation rates to optimize the route for the TSP. 

## Key Features

- **Graph Visualization**: Visualizes the graph representing cities and distances using `networkx`.
- **Evolutionary Algorithm**: Implements improved initialization, selection, crossover, and mutation strategies for optimizing the TSP.
- **Parameter Tuning**: Allows the tuning of key parameters like population size, tournament size, and mutation rates to achieve the best solution.
- **Convergence Curve**: Tracks the performance of the algorithm over time and plots the convergence curve for each combination of parameters.

## Graph Visualizations

![Graph1](Graph1.png)
![Graph2](Graph2.png)

## How It Works

1. **Graph Parsing**: The algorithm reads the city and distance information from XML files (e.g., `burma14.xml`, `brazil58.xml`) and creates a graph.
2. **Evolutionary Algorithm**:
   - **Initialization**: Random initialization of population routes.
   - **Selection**: Tournament selection to choose the best routes.
   - **Crossover**: Order crossover to combine parent routes.
   - **Mutation**: Swap mutation to introduce variety into the population.
3. **Fitness Evaluation**: Each route is evaluated based on the total travel cost.
4. **Convergence Tracking**: The algorithm tracks and plots the convergence of fitness values over iterations.

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/TSP-Evolutionary-Algorithm.git
   cd TSP-Evolutionary-Algorithm
