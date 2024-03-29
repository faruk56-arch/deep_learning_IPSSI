from flask import Flask, render_template, request
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib

# Charger le modèle RandomForestRegressor préalablement entraîné
model = joblib.load('RandomForestRegressor.pkl')

# Initialiser l'application Flask
app = Flask(__name__)

# Définir la route de la page d'accueil avec le formulaire
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Récupérer les données saisies par l'utilisateur depuis le formulaire
        num_poste = request.form['num_poste']
        lat = float(request.form['lat'])
        lon = float(request.form['lon'])
        alti = float(request.form['alti'])
        aaaammjj = int(request.form['aaaammjj'])

        # Utiliser le modèle pour faire une prédiction
        features = [[lat, lon, alti, aaaammjj]]  # Mettre les données dans le format attendu par le modèle
        rr_prediction = model.predict(features)

        # Afficher le résultat de la prédiction
        return render_template('result.html', num_poste=num_poste, rr_prediction=rr_prediction[0])

    # Afficher le formulaire si la méthode est GET
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
