import numpy as np
import pandas as pd
from minisom import MiniSom
from collections import Counter
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Wczytanie danych
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/dermatology/dermatology.data'
column_names = [
    'erythema', 'scaling', 'definite_borders', 'itching', 'koebner_phenomenon', 'polygonal_papules',
    'follicular_papules', 'oral_mucosal_involvement', 'knee_and_elbow_involvement', 'scalp_involvement',
    'family_history', 'melanin_incontinence', 'eosinophils_infiltrate', 'PNL_infiltrate', 'fibrosis_of_the_papillary_dermis',
    'exocytosis', 'acanthosis', 'hyperkeratosis', 'parakeratosis', 'clubbing_of_the_rete_ridges',
    'elongation_of_the_rete_ridges', 'thinning_of_the_suprapapillary_epidermis', 'spongiform_pustule',
    'munro_microabcess', 'focal_hypergranulosis', 'disappearance_of_the_granular_layer', 'vacuolisation_and_damage_of_basal_layer',
    'spongiosis', 'saw_tooth_appearance_of_retes', 'follicular_horn_plug', 'perifollicular_parakeratosis', 'inflammatory_monoluclear_inflitrate',
    'band_like_infiltrate', 'age', 'class'
]

data = pd.read_csv(url, header=None, names=column_names, na_values='?')

# Obsługa brakujących wartości - usuwanie linijek z brakującymi wartościami
data.dropna(inplace=True)

# Kodowanie etykiet klas (One-Hot Encoding)
class_mapping = {
    1: 'łuszczyca',
    2: 'łojotokowe zapalenie skóry',
    3: 'liszaj płaski',
    4: 'łupież różowy',
    5: 'przewlekłe zapalenie skóry',
    6: 'Łupież czerwony mieszkowy'
}
data['class'] = data['class'].map(class_mapping)
y = pd.get_dummies(data['class'])
X = data.drop(columns=['class', 'age'])

# Normalizacja danych
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Funkcja przypisywania klas do neuronów
def assign_classes_to_neurons(X_train, y_train, som):
    neuron_classes = {}
    for i, x in enumerate(X_train):
        winner = som.winner(x)
        class_label = np.argmax(y_train.iloc[i])  # Przekształcenie one-hot na indeks klasy
        if winner not in neuron_classes:
            neuron_classes[winner] = []
        neuron_classes[winner].append(class_label)

    # Przypisanie klasy na podstawie większości
    for neuron in neuron_classes:
        most_common_class = Counter(neuron_classes[neuron]).most_common(1)[0][0]
        neuron_classes[neuron] = most_common_class

    return neuron_classes

# Funkcja do przeprowadzania walidacji krzyżowej
def cross_validate_som(X, y, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    accuracies = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Obliczenie rozmiaru siatki SOM: 5 * sqrt(N)
        N = X_train.shape[0]  # Liczba wektorów danych
        grid_size = int(np.ceil(5 * np.sqrt(N)))  # Rozmiar siatki

        # Tworzenie i trenowanie SOM
        som = MiniSom(x=grid_size, y=grid_size, input_len=X_train.shape[1], sigma=0.5, learning_rate=0.3)
        som.train_random(data=X_train, num_iteration=100)

        # Przypisanie klas do neuronów
        neuron_classes = assign_classes_to_neurons(X_train, y_train, som)

        # Przewidywanie klas dla danych testowych
        y_pred_classes = []
        for x in X_test:
            winner = som.winner(x)
            y_pred_classes.append(neuron_classes.get(winner, -1))  # -1, jeśli neuron nie ma przypisanej klasy

        # Konwersja y_test do oryginalnych klas
        y_test_classes = np.argmax(y_test.to_numpy(), axis=1)

        # Obliczenie dokładności
        accuracy = np.mean(np.array(y_pred_classes) == y_test_classes) * 100
        accuracies.append(accuracy)

    return np.mean(accuracies), np.std(accuracies)

# Przeprowadzenie walidacji krzyżowej
mean_accuracy, std_accuracy = cross_validate_som(X_scaled, y)
print(f'Średnia dokładność w walidacji krzyżowej: {mean_accuracy:.2f}%')
print(f'Odchylenie Standardowe: {std_accuracy:.2f}%')
