import random
import sys
import re


class AntColonySOP:
    def __init__(self, distances, precedence, start_node, end_node,
                 n_ants=10, n_iterations=100,
                 alpha=1.0, beta=2.0, rho=0.1, q=100):
        self.distances = distances
        self.precedence = precedence
        self.start_node = start_node
        self.end_node = end_node
        self.nodes = list(distances.keys())
        self.n_nodes = len(self.nodes)
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.q = q
        self.pheromones = {i: {j: 1.0 for j in self.nodes if i != j} for i in self.nodes}
        self.best_path = []
        self.best_path_cost = float('inf')

    def solve(self):
        for i in range(self.n_iterations):
            all_paths = []
            for _ in range(self.n_ants):
                path, cost = self._construct_ant_path()
                if cost != float('inf'):
                    all_paths.append((path, cost))

            if not all_paths:
                if (i + 1) % 10 == 0:
                    print(f"Ітерація {i + 1}: Жодна мураха не змогла побудувати повний шлях.")
                continue

            best_iteration_path, best_iteration_cost = min(all_paths, key=lambda x: x[1])

            if best_iteration_cost < self.best_path_cost and best_iteration_cost > 0:
                self.best_path = best_iteration_path
                self.best_path_cost = best_iteration_cost

            self._update_pheromones()

            if (i + 1) % 10 == 0 or i == self.n_iterations - 1:
                print(f"Ітерація {i + 1}: Найкраща вартість = {self.best_path_cost:.2f}")

        return self.best_path, self.best_path_cost

    def _construct_ant_path(self):
        path = [self.start_node]
        unvisited = set(self.nodes)
        unvisited.remove(self.start_node)
        path_set = {self.start_node}

        while unvisited:
            current_node = path[-1]
            allowed_nodes = self._get_allowed_nodes(path_set, unvisited)

            if not allowed_nodes:
                return [], float('inf')  # неможливо продовжити шлях

            next_node = self._select_next_node(current_node, allowed_nodes)

            path.append(next_node)
            unvisited.remove(next_node)
            path_set.add(next_node)

        if path[-1] != self.end_node:
            return [], float('inf')

        cost = self._calculate_path_cost(path)
        return path, cost

    def _get_allowed_nodes(self, path_set, unvisited_nodes):
        allowed = []
        for node in unvisited_nodes:
            if node == self.end_node and len(unvisited_nodes) > 1:
                continue
            prerequisites = self.precedence.get(node, set())
            if prerequisites.issubset(path_set):
                allowed.append(node)

        if not allowed and self.end_node in unvisited_nodes:
            prerequisites = self.precedence.get(self.end_node, set())
            if prerequisites.issubset(path_set):
                return [self.end_node]

        return allowed

    def _select_next_node(self, current_node, allowed_nodes):
        probabilities = {}
        total_prob = 0.0
        for node in allowed_nodes:
            tau = self.pheromones[current_node][node] ** self.alpha
            eta = (1.0 / (self.distances[current_node].get(node, 1e9) + 1e-10)) ** self.beta
            prob = tau * eta
            probabilities[node] = prob
            total_prob += prob

        if total_prob == 0:
            return random.choice(allowed_nodes)

        probs_normalized = [p / total_prob for p in probabilities.values()]
        next_node = random.choices(list(probabilities.keys()), weights=probs_normalized, k=1)[0]
        return next_node

    def _update_pheromones(self):
        for i in self.nodes:
            for j in self.nodes:
                if i != j:
                    self.pheromones[i][j] *= (1 - self.rho)

        if not self.best_path or self.best_path_cost == 0 or self.best_path_cost == float('inf'):
            return

        delta_tau = self.q / self.best_path_cost
        for i in range(len(self.best_path) - 1):
            node1 = self.best_path[i]
            node2 = self.best_path[i + 1]
            self.pheromones[node1][node2] += delta_tau

    def _calculate_path_cost(self, path):
        cost = 0
        for i in range(len(path) - 1):
            dist = self.distances[path[i]].get(path[i + 1], float('inf'))
            if dist == float('inf'):
                return float('inf')
            cost += dist
        return cost


def parse_soplib_file(filename):
    with open(filename, 'r') as f:
        content = f.read()

    dimension = int(re.search(r'DIMENSION:\s*(\d+)', content).group(1))
    edge_weight_section = re.search(r'EDGE_WEIGHT_SECTION\s*([\s\S]*?)(\s*PRECEDENCE_SECTION|\s*EOF)', content)
    if not edge_weight_section:
        raise ValueError("Не знайдено EDGE_WEIGHT_SECTION у файлі.")

    weights = list(map(int, edge_weight_section.group(1).split()))
    distances = {i: {} for i in range(dimension)}

    # деякі SOPLIB файли мають трикутну матрицю
    k = 0
    for i in range(dimension):
        for j in range(dimension):
            if i == j:
                distances[i][j] = 0
            else:
                if k < len(weights):
                    w = weights[k]
                    if w <= 0:
                        distances[i][j] = float('inf')
                    else:
                        distances[i][j] = w
                    k += 1
                else:
                    distances[i][j] = float('inf')

    # прецедентні обмеження
    precedence = {i: set() for i in range(dimension)}
    precedence_section_match = re.search(r'PRECEDENCE_SECTION\s*([\s\S]*?)\s*-1', content)
    if precedence_section_match:
        pairs = list(map(int, precedence_section_match.group(1).split()))
        for i in range(0, len(pairs), 2):
            pred, succ = pairs[i] - 1, pairs[i + 1] - 1
            if pred >= 0 and succ >= 0:
                precedence[succ].add(pred)

    return distances, precedence, 0, dimension - 1, dimension


def run_test(filename):
    print(f"\n\n{'=' * 25} ЗАПУСК ТЕСТУ: {filename} {'=' * 25}")
    try:
        distances, precedence, start, end, dim = parse_soplib_file(filename)
    except FileNotFoundError:
        print(f"ПОМИЛКА: Файл '{filename}' не знайдено.")
        return
    except Exception as e:
        print(f"ПОМИЛКА при читанні '{filename}': {e}")
        return

    print(f"Розмірність задачі: {dim} вузлів")

    if dim < 30:
        iterations, ants, beta = 75, 15, 3.0
    elif dim < 80:
        iterations, ants, beta = 150, 25, 5.0
    else:
        iterations, ants, beta = 200, 30, 5.0

    aco_solver = AntColonySOP(
        distances=distances, precedence=precedence,
        start_node=start, end_node=end,
        n_ants=ants, n_iterations=iterations,
        alpha=1.0, beta=beta, rho=0.2, q=100)

    best_path, best_cost = aco_solver.solve()

    print("\n" + "-" * 40)
    print(f"РЕЗУЛЬТАТИ ДЛЯ '{filename}'")
    if best_path and best_cost != float('inf') and best_cost > 0:
        path_str = ' -> '.join(map(lambda x: str(x + 1), best_path[:15]))
        if len(best_path) > 15:
            path_str += " ..."
        print(f"Найкращий знайдений маршрут (номерація з 1): {path_str}")
        print(f"Його вартість: {best_cost:.2f}")
    else:
        print("Не вдалося знайти допустимий маршрут.")
    print("-" * 40)


if __name__ == '__main__':
    test_files = ["br17.10.sop", "ft53.1.sop", "rbg109a.sop"]
    for file in test_files:
        run_test(file)
