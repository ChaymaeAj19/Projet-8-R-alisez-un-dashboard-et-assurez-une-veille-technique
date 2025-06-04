import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import joblib
import shap
import matplotlib.pyplot as plt
import io
import base64

# --- Config page ---
st.set_page_config(page_title="Dashboard Scoring Crédit", layout="wide")
st.title("📊 Dashboard Scoring Crédit - Prototype avec SHAP local")

API_URL = "https://projet-7-implementation.onrender.com"

# --- Chargement données ---
@st.cache_data
def load_data():
    return pd.read_csv("features_for_prediction.csv")

df = load_data()

# --- Chargement modèle localement pour SHAP ---
@st.cache_resource
def load_model():
    # Chemin local vers ton pipeline LGBM picklé
    model_path = "Simulations/Best_model/lgbm_pipeline1.pkl"
    model_bundle = joblib.load(model_path)
    pipeline = model_bundle['pipeline']
    expected_features = model_bundle['features']
    model = pipeline.steps[-1][1]
    explainer = shap.TreeExplainer(model)
    return pipeline, expected_features, explainer

pipeline, expected_features, explainer = load_model()

numeric_cols = df.select_dtypes(include="number").columns.tolist()
client_ids = df["SK_ID_CURR"].unique()

# --- Sélection client ---
client_id = st.selectbox("🔎 Sélectionnez un client", client_ids)
client_data = df[df["SK_ID_CURR"] == client_id].iloc[0]

# --- Appel API pour récupérer score ---
def predict_api(client_id):
    try:
        response = requests.post(f"{API_URL}/predict", json={"SK_ID_CURR": int(client_id)})
        return response.json()
    except Exception as e:
        return {"error": str(e)}

res = predict_api(client_id)
score = res.get("probability", None)
if score is None:
    st.error("Erreur récupération score depuis API.")
    st.stop()

decision = "Accord" if score >= 50 else "Refus"

# --- Affichage score & jauge ---
col1, col2 = st.columns([1, 2])
with col1:
    st.metric("Score crédit (%)", f"{score:.2f}%")
    if decision == "Accord":
        st.success(f"Décision : {decision}")
    else:
        st.error(f"Décision : {decision}")

    # Jauge Plotly
    fig_gauge = px.bar_polar(
        r=[score, 100 - score],
        theta=["Score", "Distance au seuil"],
        color=["Score", "Distance au seuil"],
        color_discrete_sequence=px.colors.sequential.Plasma_r,
        title="Score crédit"
    )
    st.plotly_chart(fig_gauge, use_container_width=True)

with col2:
    st.markdown("### Caractéristiques du client")
    st.dataframe(client_data[expected_features])

# --- Calcul SHAP local ---
st.markdown("---")
st.markdown("## Explication du score par SHAP (local)")

# Préparer input modèle (attention à la colonne SK_ID_CURR souvent exclue)
X_client = client_data[expected_features].to_frame().T
X_client = X_client.apply(pd.to_numeric, errors='coerce').fillna(0)

shap_values = explainer.shap_values(X_client)[1]  # classe 1 (risque)

# Affichage graphique SHAP waterfall local avec matplotlib et Streamlit
fig, ax = plt.subplots(figsize=(10, 6))
shap.waterfall_plot(shap.Explanation(values=shap_values[0],
                                    base_values=explainer.expected_value[1],
                                    data=X_client.iloc[0],
                                    feature_names=expected_features), max_display=10)
st.pyplot(fig)

# --- Comparaison univariée ---
st.markdown("---")
st.markdown("## Comparaison univariée avec la population")

var_uni = st.selectbox("Variable à comparer", numeric_cols, key="uni_var")

fig_uni = px.histogram(df, x=var_uni, nbins=30, title=f"Distribution de {var_uni}")
fig_uni.add_vline(x=client_data[var_uni], line_dash="dash", line_color="red", annotation_text="Client")
st.plotly_chart(fig_uni, use_container_width=True)

# --- Analyse bivariée ---
st.markdown("---")
st.markdown("## Analyse bivariée")

var_x = st.selectbox("Feature X", numeric_cols, index=0, key="bi_var_x")
var_y = st.selectbox("Feature Y", numeric_cols, index=1, key="bi_var_y")

fig_bi = px.scatter(df, x=var_x, y=var_y, opacity=0.5,
                    title=f"Analyse bivariée : {var_x} vs {var_y}")
fig_bi.add_scatter(x=[client_data[var_x]], y=[client_data[var_y]], mode='markers',
                   marker=dict(color='red', size=15), name="Client")
st.plotly_chart(fig_bi, use_container_width=True)

# --- Modification client et recalcul ---
st.markdown("---")
st.markdown("## Modifier les informations du client")

with st.form("edit_form"):
    edited_features = {}
    for feat in expected_features:
        val = st.number_input(feat, value=float(client_data[feat]), format="%.4f")
        edited_features[feat] = val
    submit_edit = st.form_submit_button("Recalculer score")

if submit_edit:
    # On peut modifier la fonction API pour accepter dict complet de features
    try:
        response = requests.post(f"{API_URL}/predict", json={"data": edited_features})
        res_edit = response.json()
    except Exception as e:
        res_edit = {"error": str(e)}

    score_edit = res_edit.get("probability", None)
    if score_edit is not None:
        st.success(f"Score recalculé : {score_edit:.2f}%")
        decision_edit = "Accord" if score_edit >= 50 else "Refus"
        if decision_edit == "Accord":
            st.success(f"Décision : {decision_edit}")
        else:
            st.error(f"Décision : {decision_edit}")
    else:
        st.error("Erreur lors de la prédiction du score modifié.")

st.markdown("---")
st.caption("Prototype avec score, interprétabilité locale SHAP, comparaison, analyse et édition client.")
