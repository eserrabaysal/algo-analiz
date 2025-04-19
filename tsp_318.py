import numpy as np
import matplotlib.pyplot as plt
import os
import time

# --- 1. Dataset'i Oku ---
def read_dataset(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found: {file_path}")
    
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
    return np.sqrt(((coords[:, np.newaxis, :] - coords[np.newaxis, :, :]) ** 2).sum(axis=2))

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

# --- 4. Rotayı Görselleştir ve Kaydet ---
def plot_and_save_path(coords, path, cost, title, image_path):
    plt.figure(figsize=(12, 8))
    for i in range(len(path) - 1):
        plt.plot(
            [coords[path[i]][0], coords[path[i + 1]][0]],
            [coords[path[i]][1], coords[path[i + 1]][1]],
            'b-'
        )
    plt.plot(
        [coords[path[-1]][0], coords[path[0]][0]],
        [coords[path[-1]][1], coords[path[0]][1]],
        'r-'  # Başlangıç/son bağlantısı
    )
    plt.scatter(coords[:, 0], coords[:, 1], c='red', s=30)
    plt.title(f"{title}\nCost = {round(cost, 2)}")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.grid(True)
    plt.savefig(image_path)  # Görseli kaydet
    plt.close()

# --- 5. Çözümü Kaydet ---
def save_solution(path, cost, output_path):
    with open(output_path, "w") as f:
        f.write(f"Optimal Path: {path}\n")
        f.write(f"Total Cost: {cost}\n")

# --- 6. Ana Fonksiyon ---
if __name__ == "__main__":
    # Dataset yolu
    file_path = r"C:\Users\SAVLA\Desktop\algo\dataset\tsp_318_2"  # Dosya yolunu buraya girin
    
    try:
        # Dataset'i oku
        print(f"Reading dataset from: {file_path}")
        city_coords = read_dataset(file_path)
        dist_matrix = calculate_distance_matrix(city_coords)

        # Zaman ölçümü başlat
        start_time = time.time()

        # Greedy Algoritmasını Çalıştır
        path, cost = greedy_tsp(dist_matrix)

        # Zaman ölçümü bitir
        end_time = time.time()

        # Çıktıları Kaydet
        title = "Greedy Algorithm for TSP (Dataset 318)"
        image_path = "greedy_tsp_path_318.png"
        output_path = "greedy_tsp_solution_318.txt"
        plot_and_save_path(city_coords, path, cost, title, image_path)
        save_solution(path, cost, output_path)

        # Sonuçları Yazdır
        print(f"{title}\nOptimal Cost: {cost}")
        print(f"Path and cost saved to: {output_path}")
        print(f"Path visualization saved to: {image_path}")
        print(f"Execution Time: {end_time - start_time:.2f} seconds")
    
    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")