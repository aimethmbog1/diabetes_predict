import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import base64
import os
from sklearn.preprocessing import StandardScaler

# -------------------
# Fonctions utilitaires
# -------------------

@st.cache_resource(show_spinner=False)
def load_model(path='model_dump.pkl'):
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    else:
        return None

@st.cache_resource(show_spinner=False)
def load_scaler(path='scaler.pkl'):
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    else:
        # Retourne un scaler vide, non entraîné
        return StandardScaler()

@st.cache_data(show_spinner=False)
def load_data(file):
    try:
        return pd.read_csv(file)
    except Exception as e:
        st.error(f"Erreur lors du chargement du fichier: {e}")
        return None

def filedownload(df, filename="resultats_predictions.csv"):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Télécharger le fichier CSV</a>'
    return href

def validate_data(df, required_cols):
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        return False, f"Colonnes manquantes : {', '.join(missing_cols)}"
    if (df[required_cols] < 0).any().any():
        return False, "Certaines valeurs sont négatives, ce qui est impossible."
    return True, "Données valides"

# -------------------
# Fonction principale
# -------------------

def main():
    st.set_page_config(page_title="Diabetes Predictor", layout="centered")
    
    st.sidebar.title("Menu")
    menu = ["Accueil", "Analyse", "Visualisation", "Machine Learning", "Admin", "À propos"]
    choice = st.sidebar.radio("Navigation", menu)

    sidebar_img_path = "diabetes.jpg"
    if os.path.exists(sidebar_img_path):
        st.sidebar.image(sidebar_img_path)
    else:
        st.sidebar.warning("Image de la sidebar non trouvée.")

    data_path = "diabetes.csv"
    data = load_data(data_path) if os.path.exists(data_path) else None

    model = load_model()
    scaler = load_scaler()

    if choice == "Accueil":
        st.markdown("<h1 style='text-align: center; color: brown;'>Diabetes Prediction App</h1>", unsafe_allow_html=True)
        st.markdown("<h3 style='text-align: center; color: gray;'>Une étude sur le diabète au Cameroun</h3>", unsafe_allow_html=True)

        main_img_path = "images2.jpg"
        if os.path.exists(main_img_path):
            st.image(main_img_path, width=400)
        else:
            st.warning("Image principale non trouvée.")

        st.write("""
            Cette application permet d'explorer, analyser et prédire les risques de diabète à l'aide d'un modèle d'apprentissage automatique.
        """)
        st.subheader("Contexte au Cameroun")
        st.write("""
            La prévalence du diabète chez les adultes en milieu urbain au Cameroun est estimée à 6–8 %, 
            avec environ 80 % des cas non diagnostiqués. En 10 ans (1994–2004), la prévalence a été multipliée par 10.
        """)

    elif choice == "Analyse":
        st.subheader("Analyse des données")
        if data is not None:
            st.dataframe(data.head())

            if st.checkbox("Statistiques descriptives"):
                st.write(data.describe())

            if st.checkbox("Matrice de corrélation"):
                fig, ax = plt.subplots(figsize=(12, 10))
                sns.heatmap(data.corr(), annot=True, cmap="coolwarm", ax=ax)
                st.pyplot(fig)

            if st.checkbox("Afficher les colonnes"):
                st.write(data.columns.tolist())

            st.markdown(filedownload(data, filename="diabetes.csv"), unsafe_allow_html=True)
        else:
            st.error("Le fichier diabetes.csv est introuvable.")

    elif choice == "Visualisation":
        st.subheader("Visualisation des données")
        if data is not None:
            if st.checkbox("Countplot par âge"):
                fig, ax = plt.subplots(figsize=(12, 6))
                sns.countplot(x='Age', data=data, ax=ax)
                st.pyplot(fig)

            if st.checkbox("Nuage de points Glucose vs Age"):
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.scatterplot(x='Glucose', y='Age', data=data, hue='Outcome', ax=ax)
                st.pyplot(fig)
        else:
            st.error("Le fichier de données est manquant.")

    elif choice == "Machine Learning":
        st.subheader("Module de prédiction")

        uploaded_file = st.sidebar.file_uploader("Importer un fichier CSV", type=["csv"])

        if uploaded_file:
            df = load_data(uploaded_file)
            if df is not None:
                required_cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

                valid, msg = validate_data(df, required_cols)
                if not valid:
                    st.error(f"Validation des données échouée : {msg}")
                    return
                
                tab1, tab2, tab3 = st.tabs(["📄 Données", "📊 Visualisation", "🤖 Prédiction"])

                with tab1:
                    st.write("Aperçu des données importées")
                    st.write(df.head())

                with tab2:
                    if 'Glucose' in df.columns:
                        fig, ax = plt.subplots(figsize=(8, 6))
                        sns.histplot(data=df, x='Glucose', kde=True, ax=ax)
                        st.pyplot(fig)
                    else:
                        st.warning("La colonne 'Glucose' est manquante dans les données importées.")

                with tab3:
                    if model is None:
                        st.error("Le modèle prédictif est introuvable.")
                    else:
                        try:
                            X = df[required_cols].copy()

                            # Gestion du scaler non entraîné
                            try:
                                _ = scaler.mean_  # test si scaler est fit
                            except AttributeError:
                                scaler.fit(X)
                                with open("scaler.pkl", "wb") as f:
                                    pickle.dump(scaler, f)

                            X_scaled = scaler.transform(X)
                            prediction = model.predict(X_scaled)

                            result_df = pd.DataFrame(prediction, columns=["Prédiction"])
                            ndf = pd.concat([df.reset_index(drop=True), result_df], axis=1)
                            ndf["Prédiction"].replace({0: "Pas de risque", 1: "Risque de diabète"}, inplace=True)

                            st.write("Résultats des prédictions")
                            st.dataframe(ndf)

                            st.markdown(filedownload(ndf, filename="resultats_predictions.csv"), unsafe_allow_html=True)
                        except Exception as e:
                            st.error("Erreur lors de la prédiction, vérifiez le format des données.")
                            st.code(str(e))
        else:
            st.info("Veuillez importer un fichier CSV pour commencer.")

    elif choice == "Admin":
        st.subheader("Administration")

        st.write("Uploader un nouveau modèle (fichier .pkl)")
        new_model_file = st.file_uploader("Choisir un fichier modèle", type=["pkl"])
        if new_model_file:
            try:
                with open("model_dump.pkl", "wb") as f:
                    f.write(new_model_file.getbuffer())
                st.success("Nouveau modèle chargé avec succès !")
                st.experimental_rerun()
            except Exception as e:
                st.error("Erreur lors du chargement du nouveau modèle.")
                st.code(str(e))

        st.write("---")

        st.write("Uploader un nouveau scaler (fichier .pkl)")
        new_scaler_file = st.file_uploader("Choisir un fichier scaler", type=["pkl"])
        if new_scaler_file:
            try:
                with open("scaler.pkl", "wb") as f:
                    f.write(new_scaler_file.getbuffer())
                st.success("Nouveau scaler chargé avec succès !")
                st.experimental_rerun()
            except Exception as e:
                st.error("Erreur lors du chargement du nouveau scaler.")
                st.code(str(e))

    elif choice == "À propos":
        st.subheader("À propos de cette application")
        st.write("""
            Cette application a été développée pour fournir un outil interactif d'analyse des données de santé, 
            plus précisément du diabète, avec un modèle prédictif basé sur l'apprentissage automatique.
        """)

if __name__ == "__main__":
    main()
