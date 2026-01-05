# Reconnaissance de chiffres manuscrits avec Streamlit

Ce projet implémente une application de **Machine Learning** permettant de
reconnaître des chiffres manuscrits à partir d’images.

## Méthodologie
- Base de données : **MNIST**
- Modèle : **Réseau de neurones convolutifs (CNN)**
- Framework : **TensorFlow / Keras**
- Interface : **Streamlit**

## Fonctionnalités
- Entraînement automatique du modèle CNN
- Prétraitement robuste des images manuscrites
- Prédiction du chiffre avec niveau de confiance
- Interface web interactive

## Lancer l'application localement
```bash
pip install -r requirements.txt
streamlit run app.py
