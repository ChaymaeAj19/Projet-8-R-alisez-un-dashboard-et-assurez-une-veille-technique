import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from utils import get_prediction

st.set_page_config(page_title="Dashboard Crédit Client", layout="wide")

st.title("Dashboard Crédit Client")

client_id = st.text_input("Entrez l'identifiant du client")

if client_id:
    data = get_prediction(client_id)
    
    if "error" in data:
        st.error(f"Erreur : {data['error']}")
    else:
        st.subheader("Résultat de l'analyse")
        score = data["score"]
        prediction = data["prediction"]

        st.metric(label="Score Crédit", value=f"{score:.2f}")
        st.metric(label="Décision", value=prediction)

        # Jauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=score * 100,
            delta={'reference': 50},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "green" if score >= 0.5 else "red"},
                'steps': [
                    {'range': [0, 50], 'color': 'rgba(255,0,0,0.2)'},
                    {'range': [50, 100], 'color': 'rgba(0,255,0,0.2)'}
                ],
            },
            title={'text': "Score (%)"}
        ))
        st.plotly_chart(fig, use_container_width=True)

        # Affichage des caractéristiques client
        st.subheader("Caractéristiques du client")
        features = data["client_features"]
        df_client = pd.DataFrame.from_dict(features, orient="index", columns=["Valeur"])
        st.dataframe(df_client)
