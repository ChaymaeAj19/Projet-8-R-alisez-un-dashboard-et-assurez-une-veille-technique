import streamlit as st
import pandas as pd
import pickle
import shap
import matplotlib.pyplot as plt
import numpy as np

# === 1) CHARGEMENT DU MODÈLE ET DES DONNÉES ===
model = pickle.load(open('mlflow_model/model.pkl', 'rb'))
data = pd.read_csv('features_for_prediction.csv')

# Assure-toi que SK_ID_CURR est bien dans les données
if 'SK_ID_CURR' not in data.columns:
    st.error("La colonne 'SK_ID_CURR' est absente de features_for_prediction.csv")
    st.stop()

# Liste des features pour la prédiction (sans SK_ID_CURR)
expected_features = [col for col in data.columns if col != 'SK_ID_CURR']

# === 2) SIDEBAR POUR CHOISIR LE CLIENT ===
st.sidebar.header("🔍 Sélection du client")
client_id = st.sidebar.selectbox("Choisissez un ID client:", data['SK_ID_CURR'])

# === 3) TITRE ET INTRODUCTION ===
st.title("📊 Dashboard Crédit Accessible")
st.write("Ce dashboard affiche la prédiction et l'explication SHAP pour un client donné.")

if client_id not in data['SK_ID_CURR'].values:
    st.error("Client ID not found in the dataset.")
    st.stop()

# Récupérer les données du client sous forme de Series (index = variables)
client_data = data.loc[data['SK_ID_CURR'] == client_id, expected_features].iloc[0]

st.subheader("Informations actuelles du client")
st.write(client_data.to_frame(name='Valeur'))

# === 4) FORMULAIRE ÉDITABLE (Data Editor) ===

# Construire un DataFrame 2 colonnes pour edition
editable_df = pd.DataFrame({
    "Variable": client_data.index,
    "Valeur": client_data.values
})

st.subheader("✏️ Modifier les variables du client")
edited_df = st.data_editor(editable_df, num_rows="dynamic", use_container_width=True)

# === 5) Préparer les données modifiées pour la prédiction ===
data_for_pred = client_data.copy()

for _, row in edited_df.iterrows():
    var = row["Variable"]
    val = row["Valeur"]
    if var in data_for_pred.index:
        try:
            data_for_pred[var] = float(val)
        except:
            # En cas de problème de conversion, garder la valeur d'origine
            pass

# Mettre dans DataFrame 2D (une ligne) avec les colonnes attendues par le modèle
data_for_pred_df = pd.DataFrame([data_for_pred[expected_features]])

# === 6) PRÉDICTION DU RISQUE DE DÉFAUT ===
st.subheader("📈 Probabilité de défaut")
prediction = model.predict_proba(data_for_pred_df)[0][1]
st.write(f"**Probabilité de défaut :** {prediction:.3f}")
decision = "✅ Approuvé" if prediction < 0.5 else "❌ Refusé"
st.markdown(f"### {decision}")

# === 7) EXPLICATION SHAP LOCALE ===
explainer = shap.TreeExplainer(model['classifier'])
shap_values_local = explainer(data_for_pred_df)
if isinstance(shap_values_local, list):
    local_explanation = shap_values_local[1][0]
else:
    local_explanation = shap_values_local[0]

fig_local = plt.figure()
shap.waterfall_plot(local_explanation, show=False)
st.pyplot(fig_local)
st.caption("Impact des variables sur la prédiction du client modifié.")

# === 8) ANALYSE UNIVARIÉE ET BIVARIÉE ===
st.subheader("🔍 Analyse Univariée et Bivariée")

# Univariée : histogramme de la variable modifiée la plus influente (top 1 SHAP)
shap_importance = np.abs(local_explanation.values)
top_feature_idx = np.argmax(shap_importance)
top_feature = data_for_pred_df.columns[top_feature_idx]

fig_uni, ax_uni = plt.subplots()
ax_uni.hist(data[top_feature], bins=30, alpha=0.5, label='Population')
ax_uni.axvline(data_for_pred_df.iloc[0][top_feature], color='red', linestyle='--', label='Client')
ax_uni.set_title(f"Distribution univariée : {top_feature}")
ax_uni.legend()
st.pyplot(fig_uni)

# Bivariée : scatter plot avec la variable la plus influente vs une autre variable (première variable différente)
second_feature = None
for col in data_for_pred_df.columns:
    if col != top_feature:
        second_feature = col
        break

if second_feature:
    fig_bi, ax_bi = plt.subplots()
    ax_bi.scatter(data[top_feature], data[second_feature], alpha=0.3, label='Population')
    ax_bi.scatter(data_for_pred_df.iloc[0][top_feature], data_for_pred_df.iloc[0][second_feature], color='red', s=100, label='Client')
    ax_bi.set_xlabel(top_feature)
    ax_bi.set_ylabel(second_feature)
    ax_bi.set_title(f"Analyse bivariée : {top_feature} vs {second_feature}")
    ax_bi.legend()
    st.pyplot(fig_bi)
else:
    st.write("Pas assez de variables pour afficher un graphique bivarié.")

