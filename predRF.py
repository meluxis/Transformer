import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# Charger le dataset à partir d'un fichier CSV
directory = './stocks/AEE.csv'
df = pd.read_csv(directory)

# Convertir la colonne 'dt' en format datetime pour faciliter la manipulation des dates
df['Date'] = pd.to_datetime(df['Date'])


# Supprimer les lignes avec des valeurs manquantes dans la colonne 'LandAverageTemperature'
df = df.dropna(subset=['Adj Close'])

# Ajouter des colonnes 'year' et 'month' en extrayant ces informations de la colonne 'dt'
df['year'] = df['Date'].dt.year
df['month'] = df['Date'].dt.month

#Define the features and target
features = ['Open', 'Low', 'High', 'Close', 'Volume']
target = 'Adj Close'

# Créer les ensembles de caractéristiques (X) et de cibles (y)
X = df[features]
y = df[target]

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialiser le modèle RandomForest avec 100 arbres
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Entraîner le modèle sur les données d'entraînement
rf_model.fit(X_train, y_train)

# Faire des prédictions sur les données de test
rf_predictions = rf_model.predict(X_test)

# Calculer les métriques d'évaluation du modèle
rf_mae = mean_absolute_error(y_test, rf_predictions)  # Erreur absolue moyenne
rf_r2 = r2_score(y_test, rf_predictions)  # Coefficient de détermination
print(df.head())
# Afficher les métriques d'évaluation
print("\n=== Evaluation RandomForest ===")
print(f"Mean Absolute Error (MAE): {rf_mae:.2f}")
print(f"R² Score: {rf_r2:.2f}")

# Créer un DataFrame pour comparer les températures réelles et prédites
rf_results = pd.DataFrame({
    'Year': X_test['year'],
    'Month': X_test['month'],
    'PrixActuelFermetureAjustee': y_test,
    'RandomForestPredictionPrixAjustes': rf_predictions
})

# Filtrer les résultats pour ne conserver que les données à partir de l'année 2010
rf_results = rf_results[rf_results['Year'] >= 2018]

# Trier les résultats par année et par mois
rf_results = rf_results.sort_values(by=['Year', 'Month']).reset_index(drop=True)

# Limiter le nombre d'échantillons affichés
n_samples = 50  # Modifier ce nombre pour afficher plus ou moins d'échantillons
rf_results_subset = rf_results.head(n_samples)

# Afficher les premières prédictions dans un tableau formaté
print("\n=== First Predictions RandomForest from 2010 ===")
print(rf_results_subset.to_string(index=False))

# Créer un graphique pour visualiser les températures réelles et prédites
plt.figure(figsize=(14, 7))
plt.plot(
    rf_results_subset['Year'].astype(str) + '-' + rf_results_subset['Month'].astype(str).str.zfill(2),
    rf_results_subset['PrixActuelFermetureAjustee'].values,
    label='Actual Temperature',
    marker='o'
)
plt.plot(
    rf_results_subset['Year'].astype(str) + '-' + rf_results_subset['Month'].astype(str).str.zfill(2),
    rf_results_subset['RandomForestPredictedTemperature'].values,
    label='RandomForestPredictionPrixAjustes',
    marker='x'
)
plt.xlabel('Year-Month')
plt.ylabel('Prix fermeture ajjustés')
plt.title(f'Actual vs Predicted Adj price ')
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()