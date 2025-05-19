import networkx as nx
import matplotlib.pyplot as plt
from typing import List, Tuple
import numpy as np


class VisualizationService:
    def __init__(self):
        self.G = nx.Graph()
        self.pos = None
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        plt.ion()  # Включаем интерактивный режим

    def create_network(self, matrix: List[List[float]]):
        """Создает визуализацию сети из матрицы смежности"""
        self.G.clear()
        n = len(matrix)

        # Добавляем узлы
        for i in range(n):
            self.G.add_node(i, pos=(i % 3, i // 3))

        # Добавляем ребра
        for i in range(n):
            for j in range(i + 1, n):
                if matrix[i][j] > 0:
                    self.G.add_edge(i, j, weight=matrix[i][j])

        # Вычисляем позиции узлов
        self.pos = nx.spring_layout(self.G)

        # Очищаем график
        self.ax.clear()

        # Рисуем сеть
        nx.draw_networkx_nodes(
            self.G, self.pos, node_color="lightblue", node_size=500, ax=self.ax
        )
        nx.draw_networkx_edges(self.G, self.pos, width=1.0, alpha=0.5)
        nx.draw_networkx_labels(self.G, self.pos)

        # Добавляем веса ребер
        edge_labels = nx.get_edge_attributes(self.G, "weight")
        nx.draw_networkx_edge_labels(self.G, self.pos, edge_labels=edge_labels)

        plt.title("Network Visualization")
        plt.axis("off")
        plt.draw()
        plt.pause(0.1)

    def highlight_path(self, path: List[int], color: str = "red"):
        """Подсвечивает найденный путь"""
        if not path:
            return

        # Рисуем сеть заново
        self.ax.clear()
        nx.draw_networkx_nodes(
            self.G, self.pos, node_color="lightblue", node_size=500, ax=self.ax
        )
        nx.draw_networkx_edges(self.G, self.pos, width=1.0, alpha=0.5)
        nx.draw_networkx_labels(self.G, self.pos)

        # Подсвечиваем путь
        path_edges = list(zip(path[:-1], path[1:]))
        nx.draw_networkx_edges(
            self.G, self.pos, edgelist=path_edges, width=2.0, edge_color=color
        )

        # Подсвечиваем узлы пути
        nx.draw_networkx_nodes(
            self.G, self.pos, nodelist=path, node_color=color, node_size=500
        )

        edge_labels = nx.get_edge_attributes(self.G, "weight")
        nx.draw_networkx_edge_labels(self.G, self.pos, edge_labels=edge_labels)

        plt.title(f"Found path: {' → '.join(map(str, path))}")
        plt.axis("off")
        plt.draw()
        plt.pause(0.1)

    def show_population(
        self,
        population: List[List[int]],
        fitness: List[float],
        start_node: int,
        end_node: int,
    ):
        """Показывает текущую популяцию и их приспособленность"""
        self.ax.clear()

        # Рисуем сеть
        nx.draw_networkx_nodes(
            self.G, self.pos, node_color="lightblue", node_size=500, ax=self.ax
        )
        nx.draw_networkx_edges(self.G, self.pos, width=1.0, alpha=0.5)
        nx.draw_networkx_labels(self.G, self.pos)

        # Подсвечиваем начальную и конечную точки
        nx.draw_networkx_nodes(
            self.G, self.pos, nodelist=[start_node], node_color="green", node_size=500
        )
        nx.draw_networkx_nodes(
            self.G, self.pos, nodelist=[end_node], node_color="red", node_size=500
        )

        # Рисуем пути популяции
        for i, path in enumerate(population):
            full_path = [start_node] + path + [end_node]
            path_edges = list(zip(full_path[:-1], full_path[1:]))
            alpha = 0.3 + 0.7 * (
                fitness[i] / max(fitness)
            )  # Прозрачность зависит от приспособленности
            nx.draw_networkx_edges(
                self.G,
                self.pos,
                edgelist=path_edges,
                width=1.0,
                alpha=alpha,
                edge_color="blue",
            )

        edge_labels = nx.get_edge_attributes(self.G, "weight")
        nx.draw_networkx_edge_labels(self.G, self.pos, edge_labels=edge_labels)

        plt.title("Current Population")
        plt.axis("off")
        plt.draw()
        plt.pause(0.1)

    def close(self):
        """Закрывает окно визуализации"""
        plt.close(self.fig)
