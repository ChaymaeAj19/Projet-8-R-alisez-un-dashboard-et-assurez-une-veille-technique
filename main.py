import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import shap
import joblib
import os
import matplotlib.pyplot as plt

# === Configuration de la page ===
st.set_page_config(page_title="Dashboard Scoring Crédit", layout="wide")
st.title("📊 Dashboard - Scoring Crédit Client")

API_URL = "https://projet-7-implementation.onrender.com"

# === Chargement des données clients ===
@st.cache_data
def load_clients_data():
    df = pd.read_csv("features_for_prediction.csv")
    return df

df_all_clients = load_clients_data()

# === Sélection d'un client ===
client_ids = df_all_clients["SK_ID_CURR"].unique()
client_id = st.selectbox("🔎 Sélectionnez un client", client_ids)

# === Données du client sélectionné ===
data = df_all_clients[df_all_clients["SK_ID_CURR"] == client_id].iloc[0].to_dict()

# === Requête à l'API pour le score ===
def get_prediction(client_id):
    try:
        response = requests.post(
            f"{API_URL}/predict", json={"SK_ID_CURR": int(client_id)}
        )
        return response.json()
    except:
        return {"error": "Erreur de connexion à l'API"}

result = get_prediction(client_id)

# === Vérification de la réponse API ===
score = result.get("probability", None)
if score is None:
    st.error("❌ Erreur : le score n’a pas été reçu depuis l’API.")
    st.write("Réponse de l'API :", result)
    st.stop()

prediction = "Accord" if score >= 50 else "Refus"

# === Affichage du score ===
st.subheader(f"Client ID : {client_id}")
col1, col2 = st.columns([1, 2])

with col1:
    st.metric("Score de crédit (%)", f"{round(score, 2)}%")
    if prediction == "Accord":
        st.success(f"📌 Décision : {prediction}")
    else:
        st.error(f"📌 Décision : {prediction}")

    fig = px.bar_polar(
        r=[score, 100 - score],
        theta=["Score", "Distance au seuil"],
        color=["Score", "Distance au seuil"],
        color_discrete_sequence=px.colors.sequential.Plasma_r,
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown("#### 🔍 Caractéristiques principales du client")
    df_info = pd.DataFrame(data.items(), columns=["Variable", "Valeur"])
    st.dataframe(df_info, use_container_width=True)

# === Comparaison avec la population ===
st.markdown("### 📈 Comparaison avec la population")

numeric_cols = df_all_clients.select_dtypes(include="number").columns.tolist()
selected_var = st.selectbox("Variable à comparer", numeric_cols)

fig = px.histogram(
    df_all_clients,
    x=selected_var,
    nbins=30,
    title=f"Distribution de {selected_var} dans la population",
    color_discrete_sequence=["#636EFA"],
)

if selected_var in data:
    fig.add_vline(
        x=data[selected_var],
        line_dash="dash",
        line_color="red",
        annotation_text="Client",
    )
st.plotly_chart(fig, use_container_width=True)

# === SHAP local et global (hors API) ===
st.markdown("### 🔍 Explication du modèle avec SHAP")

# Chemins locaux
model_path = os.path.join("Simulations", "Best_model", "lgbm_pipeline1.pkl")
if not os.path.exists(model_path):
    st.warning("⚠️ Modèle non trouvé localement pour SHAP.")
else:
    try:
        model_bundle = joblib.load(model_path)
        pipeline = model_bundle["pipeline"]
        expected_features = model_bundle["features"]
        model = pipeline.steps[-1][1]

        booster = model.booster_ if hasattr(model, "booster_") else model
        explainer = shap.Explainer(booster)

        # Données client pour SHAP
        X_client = df_all_clients[df_all_clients["SK_ID_CURR"] == client_id][expected_features].apply(pd.to_numeric, errors="coerce").fillna(0)
        shap_values = explainer(X_client)

        st.markdown("#### 📌 Explication locale (client)")
        fig_local, ax = plt.subplots()
        shap.plots.waterfall(shap_values[0], show=False)
        st.pyplot(fig_local)

        st.markdown("#### 🌍 Explication globale (top features)")
        X_sample = df_all_clients[expected_features].sample(n=100, random_state=42).apply(pd.to_numeric, errors="coerce").fillna(0)
        global_shap_vals = explainer(X_sample)

        fig_global, ax2 = plt.subplots()
        shap.summary_plot(global_shap_vals.values, X_sample, plot_type="bar", show=False)
        st.pyplot(fig_global)

    except Exception as e:
        st.warning(f"⚠️ Erreur lors du calcul SHAP : {e}")

# === Fin ===
st.markdown("---")
st.caption("Prototype V1 - API pour score, SHAP local en local, comparaison client/population")
