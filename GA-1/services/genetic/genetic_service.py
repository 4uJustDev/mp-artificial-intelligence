from typing import List, Tuple
from services.network.network_service import NetworkService
from services.fitness.fitness_service import FitnessService
from services.population.population_service import PopulationService
from services.visualization.visualization_service import VisualizationService


class GeneticService:
    def __init__(self):
        self.network_service = NetworkService()
        self.fitness_service = FitnessService(self.network_service)
        self.population_service = PopulationService(self.network_service)
        self.visualization_service = VisualizationService()
        self.generations = 100
        self.step_mode = False
        self.visualize = True

    def set_network(self, size: int, adj_matrix: List[List[float]]) -> None:
        """Set network topology"""
        self.network_service.set_network(size, adj_matrix)
        if self.visualize:
            self.visualization_service.create_network(adj_matrix)

    def set_algorithm_params(
        self,
        pop_size: int,
        generations: int,
        mutation_rate: float,
        crossover_rate: float,
    ) -> None:
        """Set algorithm parameters"""
        self.generations = generations
        self.population_service.set_parameters(
            pop_size=pop_size,
            tournament_size=3,
            mutation_rate=mutation_rate,
            crossover_rate=crossover_rate,
        )

    def set_path_ends(self, start: int, end: int) -> None:
        """Set start and end nodes"""
        self.fitness_service.set_path_ends(start, end)

    def set_step_mode(self, mode: bool) -> None:
        """Set step mode for debugging"""
        self.step_mode = mode

    def set_visualization(self, mode: bool) -> None:
        """Enable or disable visualization"""
        self.visualize = mode

    def find_optimal_path(self, chromosome_length: int) -> Tuple[List[int], float]:
        """Find optimal path using genetic algorithm"""
        # Initialize population
        population = self.population_service.initialize_population(chromosome_length)

        # Main evolution loop
        for generation in range(self.generations):
            # Calculate fitness for current population
            fitness = self.fitness_service.calculate_population_fitness(population)

            # Visualize current population if enabled
            if self.visualize:
                self.visualization_service.show_population(
                    population,
                    fitness,
                    self.fitness_service.start_node,
                    self.fitness_service.end_node,
                )

            # Select parents
            parents = self.population_service.tournament_selection(population, fitness)

            # Create new population through crossover and mutation
            new_population = []
            for i in range(0, len(parents), 2):
                if i + 1 < len(parents):
                    child1, child2 = self.population_service.crossover(
                        parents[i], parents[i + 1]
                    )
                    new_population.extend([child1, child2])
                else:
                    new_population.append(parents[i])

            # Apply mutation
            new_population = [
                self.population_service.mutate(chromo) for chromo in new_population
            ]

            # Calculate fitness for new population
            new_fitness = self.fitness_service.calculate_population_fitness(
                new_population
            )

            # Combine and reduce population
            population, fitness = self.population_service.reduce_population(
                population, fitness, new_population, new_fitness
            )

            # Visualize best path if enabled
            if self.visualize:
                best_idx = fitness.index(max(fitness))
                best_path = (
                    [self.fitness_service.start_node]
                    + population[best_idx]
                    + [self.fitness_service.end_node]
                )
                self.visualization_service.highlight_path(best_path)

            # Step mode pause
            if self.step_mode:
                input("Press Enter to continue...")

        # Find and return best path
        best_idx = fitness.index(max(fitness))
        best_path = (
            [self.fitness_service.start_node]
            + population[best_idx]
            + [self.fitness_service.end_node]
        )
        best_length = 1.0 / max(fitness) if max(fitness) > 0 else float("inf")

        # Final visualization of best path
        if self.visualize:
            self.visualization_service.highlight_path(best_path, color="green")
            input("Press Enter to close visualization...")
            self.visualization_service.close()

        return best_path, best_length

    def _print_generation_info(
        self, generation: int, population: List[List[int]], fitness: List[float]
    ) -> None:
        """Print detailed generation information in step mode"""
        print(f"\nGeneration {generation}:")
        print("Population:")
        for i, (chromo, fit) in enumerate(zip(population, fitness)):
            print(f"{i}: {chromo} (fitness: {fit:.4f})")
        input("\nPress Enter to continue...")
