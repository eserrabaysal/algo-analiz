import numpy as np
import matplotlib.pyplot as plt

# --- 1. Veri Seti ---
city_coords = np.array([
    [3099, 173], [2178, 978], [138, 1610], [2082, 1753], [2302, 1127],
    [805, 272], [22, 1617], [3213, 1085], [99, 536], [1533, 1780],
    [3564, 676], [29, 6], [3808, 1375], [2221, 291], [3499, 1885],
    [3124, 408], [781, 671], [1027, 1041], [3249, 378], [3297, 491],
    [213, 220], [721, 186], [3736, 1542], [868, 731], [960, 303],
    [3825, 1101], [2779, 435], [201, 693], [2502, 1274], [765, 833],
    [3105, 1823], [1937, 1400], [3364, 1498], [3702, 1624], [2164, 1874],
    [3019, 189], [3098, 1594], [3239, 1376], [3359, 1693], [2081, 1011],
    [1398, 1100], [618, 1953], [1878, 59], [3803, 886], [397, 1217],
    [3035, 152], [2502, 146], [3230, 380], [3479, 1023], [958, 1670],
    [3423, 1241], [78, 1066], [96, 691], [3431, 78], [2053, 1461],
    [3048, 1], [571, 1711], [3393, 782], [2835, 1472], [144, 1185],
    [923, 108], [989, 1997], [3061, 1211], [2977, 39], [1668, 658],
    [878, 715], [678, 1599], [1086, 868], [640, 110], [3551, 1673],
    [106, 1267], [2243, 1332], [3796, 1401], [2643, 1320], [48, 267],
    [1357, 1905], [2650, 802], [1774, 107], [1307, 964], [3806, 746],
    [2687, 1353], [43, 1957], [3092, 1668], [185, 1542], [834, 629],
    [40, 462], [1183, 1391], [2048, 1628], [1097, 643], [1838, 1732],
    [234, 1118], [3314, 1881], [737, 1285], [779, 777], [2312, 1949],
    [2576, 189], [3078, 1541], [2781, 478], [705, 1812], [3409, 1917],
    [323, 1714], [1660, 1556], [3729, 1188], [693, 1383], [2361, 640],
    [2433, 1538], [554, 1825], [913, 317], [3586, 1909], [2636, 727],
    [1000, 457], [482, 1337], [3704, 1082], [3635, 1174], [1362, 1526],
    [2049, 417], [2552, 1909], [3939, 640], [219, 898], [812, 351],
    [901, 1552], [2513, 1572], [242, 584], [826, 1226], [3278, 799],
    [86, 1065], [14, 454], [1327, 1893], [2773, 1286], [2469, 1838],
    [3835, 963], [1031, 428], [3853, 1712], [1868, 197], [1544, 863],
    [457, 1607], [3174, 1064], [192, 1004], [2318, 1925], [2232, 1374],
    [396, 828], [2365, 1649], [2499, 658], [1410, 307], [2990, 214],
    [3646, 1018], [3394, 1028], [1779, 90], [1058, 372], [2933, 1459],
])

num_cities = len(city_coords)

# --- 2. Mesafe Matrisi ---
dist_matrix = np.linalg.norm(city_coords[:, None, :] - city_coords[None, :, :], axis=-1)

# --- 3. Toplam Mesafe Fonksiyonu ---
def total_distance(path):
    return sum(dist_matrix[path[i], path[i + 1]] for i in range(len(path) - 1)) + dist_matrix[path[-1], path[0]]

# --- 4. 2-Opt İyileştirme ---
def two_opt(route):
    best = route
    improved = True
    while improved:
        improved = False
        for i in range(1, len(route) - 2):
            for j in range(i + 1, len(route)):
                if j - i == 1: continue
                new_route = route[:i] + route[i:j][::-1] + route[j:]
                if total_distance(new_route) < total_distance(best):
                    best = new_route
                    improved = True
        route = best
    return best

# --- 5. Karınca Kolonisi Optimizasyonu ---
def ant_colony_optimization(dist_matrix, num_ants, num_iterations, alpha, beta, evaporation_rate):
    pheromone = np.ones(dist_matrix.shape)
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
            cost = total_distance(path)
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

# --- 6. Algoritmayı Çalıştır ---
num_ants = 50
num_iterations = 100  # İterasyon sayısı azaltıldı
alpha = 1.0
beta = 2.0
evaporation_rate = 0.5

# Karınca Kolonisi Optimizasyonu ile çözüm bul
aco_path, aco_cost = ant_colony_optimization(dist_matrix, num_ants, num_iterations, alpha, beta, evaporation_rate)

# 2-Opt ile iyileştirme yap
optimal_path = two_opt(aco_path)
optimal_cost = total_distance(optimal_path)

# --- 7. Sonuçları Görselleştir ---
def plot_path(coords, path, cost):
    plt.figure(figsize=(10, 6))
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
    plt.scatter(coords[:, 0], coords[:, 1], c='red', s=50)
    plt.title(f"Optimal Path with Cost = {round(cost, 2)}")
    plt.show()

plot_path(city_coords, optimal_path, optimal_cost)

# --- 8. Sonuçları Kaydet ---
with open("optimal_path_hybrid.txt", "w") as f:
    f.write(f"Optimal Path: {' -> '.join(map(str, optimal_path))}\n")
    f.write(f"Optimal Cost: {round(optimal_cost, 2)}\n")

print("Sonuç 'optimal_path_hybrid.txt' dosyasına kaydedildi.")