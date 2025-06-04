import streamlit as st
import pandas as pd
import requests
import plotly.express as px

API_URL = "https://projet-7-implementation.onrender.com"

# === Chargement des données client ===
@st.cache_data
def load_data():
    return pd.read_csv("features_for_prediction.csv")

df = load_data()
client_ids = df["SK_ID_CURR"].tolist()

st.title("📊 Dashboard Crédit Client")

# === Sélection du client ===
client_id = st.selectbox("Sélectionner un ID client :", client_ids)

# === Affichage des infos de base ===
client_data = df[df["SK_ID_CURR"] == client_id]

st.subheader("🔍 Informations du client")
st.dataframe(client_data.T, use_container_width=True)

# === Appel API : Score de crédit ===
st.subheader("🎯 Score de Crédit")

try:
    response = requests.post(f"{API_URL}/predict", json={"SK_ID_CURR": int(client_id)})
    result = response.json()
    if "probability" in result:
        probability = result["probability"]
        st.metric("Probabilité d'acceptation", f"{probability} %", delta=None)
        
        # Jauge visuelle
        st.progress(probability / 100)

        # Explication textuelle générique
        if probability >= 50:
            st.success("Le client est probablement éligible au crédit.")
        else:
            st.error("Le client risque d'être refusé. Vérifiez les facteurs contributifs.")

    else:
        st.warning("⚠️ Score non disponible pour ce client.")

except Exception as e:
    st.error(f"Erreur d'appel API : {str(e)}")

# === Comparaison aux autres clients ===
st.subheader("📊 Comparaison avec les autres clients")

col_to_compare = st.selectbox("Choisir une variable à comparer :", df.columns[1:])

fig = px.histogram(df, x=col_to_compare, nbins=50, title=f"Distribution de {col_to_compare}")
fig.add_vline(x=client_data[col_to_compare].values[0], line_dash="dash", line_color="red")
st.plotly_chart(fig, use_container_width=True)

# === (Optionnel) Affichage du SHAP si dispo sous forme textuelle ===
if st.checkbox("Afficher les explications SHAP (version texte si disponible)"):
    try:
        shap_response = requests.post(f"{API_URL}/predict", json={"SK_ID_CURR": int(client_id), "with_shap": True})
        shap_result = shap_response.json()
        if "shap_plot_base64" in shap_result:
            st.image(f"data:image/png;base64,{shap_result['shap_plot_base64']}", use_column_width=True)
        else:
            st.info("Explication graphique SHAP indisponible pour ce client (limite mémoire ?)")
    except:
        st.warning("Erreur lors de la récupération des explications SHAP.")

# === (Bonus) SHAP Global ===
if st.checkbox("Afficher l'importance globale des variables (SHAP global)"):
    try:
        shap_global = requests.get(f"{API_URL}/shap_global").json()
        if "image" in shap_global:
            st.image(f"data:image/png;base64,{shap_global['image']}", caption="SHAP Global")
        else:
            st.warning("Image SHAP globale indisponible.")
    except:
        st.warning("Erreur lors de l'appel à l'API SHAP global.")

