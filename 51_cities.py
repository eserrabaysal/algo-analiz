import numpy as np
from scipy.optimize import dual_annealing
from scipy.spatial.distance import cdist

# --- 1. Şehir koordinatları ---
city_coords = np.array([
    [27, 68], [30, 48], [43, 67], [58, 48], [58, 27],
    [37, 69], [38, 46], [46, 10], [61, 33], [62, 63],
    [63, 69], [32, 22], [45, 35], [59, 15], [5, 6],
    [10, 17], [21, 10], [5, 64], [30, 15], [39, 10],
    [32, 39], [25, 32], [25, 55], [48, 28], [56, 37],
    [30, 40], [37, 52], [49, 49], [52, 64], [20, 26],
    [40, 30], [21, 47], [17, 63], [31, 62], [52, 33],
    [51, 21], [42, 41], [31, 32], [5, 25], [12, 42],
    [36, 16], [52, 41], [27, 23], [17, 33], [13, 13],
    [57, 58], [62, 42], [42, 57], [16, 57], [8, 52],
    [7, 38]
])

num_cities = len(city_coords)

# --- 2. Mesafe Matrisi ---
dist_matrix = cdist(city_coords, city_coords)

# --- 3. Hedef Fonksiyon (Toplam mesafeyi hesaplar) ---
def total_distance(order):
    order = np.argsort(order)  # Sürekli değerleri sıralamaya çevir
    total = 0
    for i in range(num_cities - 1):
        total += dist_matrix[order[i], order[i+1]]
    total += dist_matrix[order[-1], order[0]]  # Başlangıç şehrine dönüş
    return total

# Çalıştırmaların sonuçlarını yazdırmak için
results = []
costs = []  # Optimal maliyetleri saklamak için liste

# 10 kez çalıştır ve sonuçları kaydet
for i in range(10):
    bounds = [(0, 1)] * num_cities  # Her şehir için bir değer
    result = dual_annealing(total_distance, bounds, maxiter=1000)

    optimal_order = np.argsort(result.x)
    optimal_path = list(optimal_order) + [optimal_order[0]]
    optimal_cost = total_distance(result.x)

    # Sonuçları kaydet
    results.append((optimal_path, round(optimal_cost, 2)))
    costs.append(optimal_cost)  # Maliyeti listeye ekle

# Ortalama maliyeti hesapla
average_cost = np.mean(costs)

# Sonuçları bir dosyaya yazalım
file_path = "optimal_tsp_results_51_cities.txt"

with open(file_path, "w") as file:
    for i, (path, cost) in enumerate(results):
        file.write(f"Run {i+1}:\n")
        file.write(f"Optimal Path: {path}\n")
        file.write(f"Optimal Cost: {cost}\n")
        file.write("\n")
    
    # Ortalama maliyeti de dosyaya ekle
    file.write(f"Average Optimal Cost: {round(average_cost, 2)}\n")

print(f"Sonuçlar {file_path} dosyasına yazıldı.")
