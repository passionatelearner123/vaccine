"""
Enhanced Meerkat Optimization Algorithm (EMOA) for hyperparameter tuning
"""
import numpy as np
import random
import torch
from typing import Dict, List, Tuple, Callable
from config import Config

class EMOA:
    """
    Enhanced Meerkat Optimization Algorithm with crossover and dynamic re-initialization
    """
    
    def __init__(self, objective_function: Callable, search_space: Dict,
                 population_size: int = 30, max_iterations: int = 50,
                 crossover_rate: float = 0.8, mutation_rate: float = 0.2,
                 reinitialization_threshold: float = 0.1):
        """
        Initialize EMOA optimizer
        
        Args:
            objective_function: Function to optimize (should return fitness score)
            search_space: Dictionary defining hyperparameter search space
            population_size: Number of meerkats in the population
            max_iterations: Maximum number of iterations
            crossover_rate: Probability of crossover operation
            mutation_rate: Probability of mutation operation
            reinitialization_threshold: Threshold for dynamic re-initialization
        """
        self.objective_function = objective_function
        self.search_space = search_space
        self.population_size = population_size
        self.max_iterations = max_iterations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.reinitialization_threshold = reinitialization_threshold
        
        self.population = []
        self.fitness_scores = []
        self.best_solution = None
        self.best_fitness = float('-inf')
        self.convergence_history = []
        
    def _initialize_population(self):
        """Initialize population with random hyperparameters"""
        self.population = []
        
        for _ in range(self.population_size):
            individual = {}
            for param_name, param_range in self.search_space.items():
                if isinstance(param_range, list):
                    # Discrete values
                    individual[param_name] = random.choice(param_range)
                elif isinstance(param_range, tuple):
                    # Continuous range
                    individual[param_name] = random.uniform(param_range[0], param_range[1])
                else:
                    individual[param_name] = param_range
            
            self.population.append(individual)
    
    def _evaluate_population(self):
        """Evaluate fitness of all individuals in population"""
        self.fitness_scores = []
        
        for individual in self.population:
            try:
                fitness = self.objective_function(individual)
                self.fitness_scores.append(fitness)
                
                # Update best solution
                if fitness > self.best_fitness:
                    self.best_fitness = fitness
                    self.best_solution = individual.copy()
            except Exception as e:
                print(f"Error evaluating individual: {e}")
                self.fitness_scores.append(float('-inf'))
        
        self.convergence_history.append(self.best_fitness)
    
    def _crossover(self, parent1: Dict, parent2: Dict) -> Tuple[Dict, Dict]:
        """Perform crossover operation between two parents"""
        if random.random() > self.crossover_rate:
            return parent1.copy(), parent2.copy()
        
        child1 = parent1.copy()
        child2 = parent2.copy()
        
        # Select random parameters to crossover
        params_to_cross = random.sample(
            list(self.search_space.keys()),
            k=random.randint(1, len(self.search_space) // 2)
        )
        
        for param in params_to_cross:
            child1[param], child2[param] = child2[param], child1[param]
        
        return child1, child2
    
    def _mutate(self, individual: Dict) -> Dict:
        """Perform mutation operation on individual"""
        mutated = individual.copy()
        
        for param_name, param_range in self.search_space.items():
            if random.random() < self.mutation_rate:
                if isinstance(param_range, list):
                    # Discrete mutation
                    mutated[param_name] = random.choice(param_range)
                elif isinstance(param_range, tuple):
                    # Continuous mutation with Gaussian noise
                    current_value = mutated[param_name]
                    std = (param_range[1] - param_range[0]) * 0.1
                    new_value = current_value + random.gauss(0, std)
                    # Clip to bounds
                    mutated[param_name] = max(param_range[0], min(param_range[1], new_value))
        
        return mutated
    
    def _meerkat_behavior(self, individual: Dict, fitness: float) -> Dict:
        """
        Simulate meerkat behavior: sentinel, foraging, and alert
        """
        new_individual = individual.copy()
        
        # Sentinel behavior: explore around current position
        if random.random() < 0.3:
            for param_name, param_range in self.search_space.items():
                if isinstance(param_range, tuple):
                    current_value = new_individual[param_name]
                    exploration_range = (param_range[1] - param_range[0]) * 0.05
                    new_value = current_value + random.uniform(-exploration_range, exploration_range)
                    new_individual[param_name] = max(param_range[0], min(param_range[1], new_value))
        
        # Foraging behavior: move towards better regions
        if fitness < self.best_fitness and random.random() < 0.5:
            # Move towards best solution
            for param_name in self.search_space.keys():
                if isinstance(self.search_space[param_name], tuple):
                    best_value = self.best_solution[param_name]
                    current_value = new_individual[param_name]
                    # Move 20% towards best
                    new_individual[param_name] = current_value + 0.2 * (best_value - current_value)
        
        return new_individual
    
    def _dynamic_reinitialization(self):
        """Dynamically re-initialize stagnant individuals"""
        if len(self.convergence_history) < 5:
            return
        
        # Check if convergence has stalled
        recent_improvement = self.convergence_history[-1] - self.convergence_history[-5]
        
        if abs(recent_improvement) < self.reinitialization_threshold:
            # Re-initialize worst performing individuals
            sorted_indices = np.argsort(self.fitness_scores)
            num_to_reinit = int(self.population_size * 0.2)  # Re-initialize 20%
            
            for idx in sorted_indices[:num_to_reinit]:
                # Create new random individual
                new_individual = {}
                for param_name, param_range in self.search_space.items():
                    if isinstance(param_range, list):
                        new_individual[param_name] = random.choice(param_range)
                    elif isinstance(param_range, tuple):
                        new_individual[param_name] = random.uniform(param_range[0], param_range[1])
                
                self.population[idx] = new_individual
    
    def optimize(self) -> Dict:
        """Run EMOA optimization"""
        print("Initializing EMOA population...")
        self._initialize_population()
        
        print("Evaluating initial population...")
        self._evaluate_population()
        
        for iteration in range(self.max_iterations):
            print(f"\nIteration {iteration + 1}/{self.max_iterations}")
            print(f"Best fitness so far: {self.best_fitness:.4f}")
            
            new_population = []
            
            # Selection and reproduction
            for i in range(0, self.population_size, 2):
                # Tournament selection
                tournament_size = 3
                tournament_indices = random.sample(range(self.population_size), tournament_size)
                tournament_fitness = [self.fitness_scores[idx] for idx in tournament_indices]
                winner_idx = tournament_indices[np.argmax(tournament_fitness)]
                parent1 = self.population[winner_idx]
                
                # Select second parent
                tournament_indices = random.sample(range(self.population_size), tournament_size)
                tournament_fitness = [self.fitness_scores[idx] for idx in tournament_indices]
                winner_idx = tournament_indices[np.argmax(tournament_fitness)]
                parent2 = self.population[winner_idx]
                
                # Crossover
                child1, child2 = self._crossover(parent1, parent2)
                
                # Mutation
                child1 = self._mutate(child1)
                child2 = self._mutate(child2)
                
                # Meerkat behavior
                child1 = self._meerkat_behavior(child1, self.fitness_scores[winner_idx])
                child2 = self._meerkat_behavior(child2, self.fitness_scores[winner_idx])
                
                new_population.extend([child1, child2])
            
            # Update population
            self.population = new_population[:self.population_size]
            
            # Evaluate new population
            self._evaluate_population()
            
            # Dynamic re-initialization
            self._dynamic_reinitialization()
        
        print(f"\nOptimization complete!")
        print(f"Best fitness: {self.best_fitness:.4f}")
        print(f"Best hyperparameters: {self.best_solution}")
        
        return self.best_solution

