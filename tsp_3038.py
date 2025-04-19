import numpy as np
import matplotlib.pyplot as plt

# --- 1. Dataset'i Oku ---
def read_dataset(file_path):
    with open(file_path, "r") as f:
        lines = f.readlines()
    num_cities = int(lines[0].strip())  # İlk satır şehir sayısını içerir
    city_coords = []
    for line in lines[1:]:
        x, y = map(float, line.strip().split())
        city_coords.append((x, y))
    return np.array(city_coords)

# --- 2. Mesafe Matrisi ---
def calculate_distance_matrix(coords):
    num_cities = coords.shape[0]
    dist_matrix = np.zeros((num_cities, num_cities))
    for i in range(num_cities):
        for j in range(num_cities):
            dist_matrix[i, j] = np.linalg.norm(coords[i] - coords[j])
    return dist_matrix

# --- 3. Greedy Algoritması ---
def greedy_tsp(dist_matrix):
    num_cities = dist_matrix.shape[0]
    visited = [False] * num_cities
    path = [0]  # Başlangıç şehri (0)
    visited[0] = True
    total_cost = 0

    for _ in range(num_cities - 1):
        current_city = path[-1]
        next_city = np.argmin([dist_matrix[current_city][j] if not visited[j] else float("inf") for j in range(num_cities)])
        total_cost += dist_matrix[current_city][next_city]
        path.append(next_city)
        visited[next_city] = True

    # Başlangıç şehrine dönüş
    total_cost += dist_matrix[path[-1]][path[0]]
    path.append(0)
    return path, total_cost

# --- 4. Ant Colony Optimization ---
def ant_colony_optimization(dist_matrix, num_ants, num_iterations, alpha, beta, evaporation_rate):
    num_cities = dist_matrix.shape[0]
    pheromone = np.ones((num_cities, num_cities))  # Feromon izleri
    best_path = None
    best_cost = float("inf")
    
    for iteration in range(num_iterations):
        all_paths = []
        all_costs = []
        
        for ant in range(num_ants):
            path = [np.random.randint(num_cities)]
            while len(path) < num_cities:
                current_city = path[-1]
                probabilities = []
                for next_city in range(num_cities):
                    if next_city not in path:
                        tau = pheromone[current_city, next_city] ** alpha
                        eta = (1.0 / dist_matrix[current_city, next_city]) ** beta
                        probabilities.append(tau * eta)
                    else:
                        probabilities.append(0)
                probabilities = np.array(probabilities)
                probabilities /= probabilities.sum()
                next_city = np.random.choice(range(num_cities), p=probabilities)
                path.append(next_city)
            path.append(path[0])  # Başlangıç şehrine dönüş
            cost = total_distance(path, dist_matrix)
            all_paths.append(path)
            all_costs.append(cost)
            
            if cost < best_cost:
                best_path = path
                best_cost = cost
        
        # Feromon güncelleme
        pheromone *= (1 - evaporation_rate)
        for path, cost in zip(all_paths, all_costs):
            for i in range(len(path) - 1):
                pheromone[path[i], path[i + 1]] += 1.0 / cost
                pheromone[path[i + 1], path[i]] += 1.0 / cost
        
        # İlerleme durumu
        if iteration % 10 == 0:
            print(f"Iteration {iteration}: Best Cost = {best_cost}")
    
    return best_path, best_cost

# --- 5. Toplam Mesafe Hesaplama ---
def total_distance(path, dist_matrix):
    return sum(dist_matrix[path[i], path[i + 1]] for i in range(len(path) - 1))

# --- 6. Rotayı Görselleştir ve Kaydet ---
def plot_and_save_path(coords, path, cost, filename="tsp_result", title=""):
    plt.figure(figsize=(10, 6))
    for i in range(len(path) - 1):
        plt.plot(
            [coords[path[i]][0], coords[path[i + 1]][0]],
            [coords[path[i]][1], coords[path[i + 1]][1]],
            'g-',  # Yeşil çizgi rotayı belirtir
        )
    plt.plot(
        [coords[path[-1]][0], coords[path[0]][0]],
        [coords[path[-1]][1], coords[path[0]][1]],
        'r-'  # Başlangıç/son bağlantısı
    )
    plt.scatter(coords[:, 0], coords[:, 1], c='red', s=50)
    plt.title(f"{title}\nCost = {round(cost, 2)}")
    plt.savefig(f"{filename}.png")  # Görselleştirme PNG olarak kaydedilir
    plt.show()

# --- 7. Ana Fonksiyon ---
if __name__ == "__main__":
    file_path = r"C:\Users\SAVLA\Desktop\algo\dataset\tsp_3038_1"  # Dataset dosyasının yolu
    city_coords = read_dataset(file_path)
    dist_matrix = calculate_distance_matrix(city_coords)

    # Kullanıcı Seçimi: Greedy veya ACO
    print("TSP Algoritmaları: \n1. Greedy\n2. Ant Colony Optimization (ACO)")
    choice = int(input("Bir algoritma seçin (1 veya 2): "))

    if choice == 1:
        # Greedy Algoritması
        path, cost = greedy_tsp(dist_matrix)
        plot_and_save_path(city_coords, path, cost, filename="greedy_result", title="Greedy Algorithm")
        print(f"Greedy Algoritması ile Optimal Cost: {cost}")

    elif choice == 2:
        # ACO Parametreleri
        num_ants = 30
        num_iterations = 50
        alpha = 1.0
        beta = 2.0
        evaporation_rate = 0.5

        # ACO Algoritmasını Çalıştır
        path, cost = ant_colony_optimization(dist_matrix, num_ants, num_iterations, alpha, beta, evaporation_rate)
        plot_and_save_path(city_coords, path, cost, filename="aco_result", title="Ant Colony Optimization (ACO)")
        print(f"ACO ile Optimal Cost: {cost}")
    else:
        print("Geçersiz seçim. Lütfen 1 veya 2 seçin.")
        
    # --- Optimal Yolu Kaydet ---
    optimal_path_coords = [city_coords[city] for city in path]

    result_filename = "tsp_result.txt"
    with open(result_filename, "w") as f:
        f.write("--- TSP Çözüm Sonuçları ---\n")
        f.write(f"Algoritma: {'Greedy' if choice == 1 else 'Ant Colony Optimization (ACO)'}\n")
        f.write(f"Toplam Maliyet (Cost): {round(cost, 2)}\n")
        f.write(f"Toplam Şehir Sayısı: {len(city_coords)}\n")
        f.write(f"Ziyaret Edilen Rota (Şehir İndeksleri):\n")
        f.write(" -> ".join(map(str, path)) + "\n")
        f.write("\nZiyaret Edilen Rota (Koordinatlar):\n")
        for i, coord in enumerate(optimal_path_coords):
            f.write(f"{i+1}. Şehir Koordinatları: {coord}\n")
    print(f"Sonuçlar '{result_filename}' dosyasına yazıldı.")
    print("Görselleştirme tamamlandı. PNG formatında kaydedildi.")