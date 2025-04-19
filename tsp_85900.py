import numpy as np
import matplotlib.pyplot as plt
import os
import time
from sklearn.cluster import KMeans
from multiprocessing import Pool, cpu_count

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

# --- 2. Mesafe Hesapla (Vectorized) ---
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

# --- 4. Cluster'lar İçin TSP ---
def solve_cluster(cluster_coords):
    # Mesafe matrisini hesapla
    dist_matrix = calculate_distance_matrix(cluster_coords)
    # Greedy TSP çalıştır
    path, cost = greedy_tsp(dist_matrix)
    return path, cost

# --- 5. Çoklu İşlem ---
def parallel_tsp(coords, n_clusters):
    # Kümeleri oluştur
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(coords)
    labels = kmeans.labels_
    clustered_coords = [coords[labels == i] for i in range(n_clusters)]

    # Çoklu işlem havuzu
    with Pool(cpu_count()) as pool:
        results = pool.map(solve_cluster, clustered_coords)
    
    return results

# --- 6. Rotayı Görselleştir ve Kaydet ---
def plot_and_save_path(coords, path, cost, title, image_path):
    plt.figure(figsize=(12, 8))
    for i in range(len(path) - 1):
        plt.plot(
            [coords[path[i]][0], coords[path[i + 1]][0]],
            [coords[path[i]][1], coords[path[i + 1]][1]],
            'b-'
        )
    plt.scatter(coords[:, 0], coords[:, 1], c='red', s=10, label="Cities")
    plt.scatter(coords[path[0]][0], coords[path[0]][1], c='green', s=50, label="Start/End", edgecolors="black")
    plt.title(f"{title}\nCost = {round(cost, 2)}")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.legend()
    plt.grid(True)
    plt.savefig(image_path)  # Görseli kaydet
    plt.close()

# --- 7. Ana Fonksiyon ---
if __name__ == "__main__":
    # Dataset yolu
    file_path = r"C:\Users\SAVLA\Desktop\algo\dataset\tsp_85900_1"

    try:
        # Dataset'i oku
        print(f"Reading dataset from: {file_path}")
        city_coords = read_dataset(file_path)

        # Zaman ölçümü başlat
        start_time = time.time()

        # Cluster'lı TSP çözümü
        n_clusters = 100  # 100 cluster'a böl
        cluster_results = parallel_tsp(city_coords, n_clusters)

        # Cluster sonuçlarını birleştir
        total_cost = sum([result[1] for result in cluster_results])

        # Zaman ölçümü bitir
        end_time = time.time()

        # Çıktıları Kaydet
        title = "Clustered Greedy Algorithm for TSP (Large Dataset)"
        image_path = os.path.splitext(file_path)[0] + "_clustered_path.png"
        output_path = os.path.splitext(file_path)[0] + "_clustered_solution.txt"

        # Görselleştirme (ilk cluster için)
        first_cluster_coords = city_coords[:len(cluster_results[0][0])]
        plot_and_save_path(first_cluster_coords, cluster_results[0][0], cluster_results[0][1], title, image_path)

        # Sonuçları Yazdır
        print(f"{title}\nTotal Cost: {total_cost}")
        print(f"Path visualization saved to: {image_path}")
        print(f"Execution Time: {end_time - start_time:.2f} seconds")
    
    except FileNotFoundError:
        print(f"Dataset file not found: {file_path}. Please ensure the file exists and the path is correct.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")