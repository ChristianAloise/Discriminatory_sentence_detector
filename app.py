# app.py
import streamlit as st
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
from sentence_transformers import SentenceTransformer
import os

# --- Configurazioni e Costanti ---
MODEL_FILENAME = "simple_nn_classifier.h5"
SCALER_FILENAME = "embedding_scaler.pkl"

SBERT_MODEL_NAME = 'all-mpnet-base-v2'

PREDICTION_THRESHOLD = 0.5


@st.cache_resource
def load_sbert_model():
    print(f"Caricamento modello SBERT: {SBERT_MODEL_NAME}")
    try:
        model = SentenceTransformer(SBERT_MODEL_NAME)
        print("Modello SBERT caricato.")
        return model
    except Exception as e:
        st.error(f"Errore nel caricamento del modello SBERT: {e}")
        return None

@st.cache_resource
def load_scaler(scaler_path):
    print(f"Caricamento scaler: {scaler_path}")
    try:
        scaler = joblib.load(scaler_path)
        print("Scaler caricato.")
        return scaler
    except Exception as e:
        st.error(f"Errore nel caricamento dello scaler: {e}")
        return None

@st.cache_resource
def load_keras_classifier(_model_path): # _model_path per evitare conflitti con l'oggetto modello SBERT
    print(f"Caricamento modello Keras: {_model_path}")
    try:
        model = load_model(_model_path)
        print("Modello Keras caricato.")
        return model
    except Exception as e:
        st.error(f"Errore nel caricamento del modello Keras: {e}")
        return None

# --- Funzione di Preprocessing del Testo ---
def preprocess_text_for_sbert(text):
    if not isinstance(text, str):
        text = str(text)
    return text.lower()

# --- Funzione di Predizione ---
def predict(sentence_text, sbert_model, scaler, keras_model):
    if not sentence_text or not sentence_text.strip():
        return None, 0.0, "Per favore, inserisci una frase."

    if sbert_model is None or scaler is None or keras_model is None:
        return None, 0.0, "Errore: Uno o più modelli non sono stati caricati correttamente."

    try:
        preprocessed_text = preprocess_text_for_sbert(sentence_text)
        embedding = sbert_model.encode([preprocessed_text])
        scaled_embedding = scaler.transform(embedding)
        probability = keras_model.predict(scaled_embedding, verbose=0)[0][0] # verbose=0 per pulire l'output

        if probability >= PREDICTION_THRESHOLD:
            label = "Discriminatoria"
        else:
            label = "Non Discriminatoria"
        return label, float(probability), None
    except Exception as e:
        return None, 0.0, f"Errore durante la predizione: {e}"

# --- Interfaccia Utente Streamlit ---
st.set_page_config(page_title="Rilevatore Frasi Discriminatorie", layout="wide")

st.title("⚖️ Rilevatore di Frasi Potenzialmente Discriminatorie")
st.markdown("""
Questa applicazione utilizza un modello di Intelligenza Artificiale per identificare se una frase
inserita potrebbe contenere linguaggio discriminatorio.
**Disclaimer:** *Questo è uno strumento sperimentale. Le predizioni potrebbero non essere sempre accurate
e non dovrebbero essere usate come unica base per giudizi definitivi.*
""")

# Carica i modelli all'avvio dell'app
# Verifica che i file esistano prima di tentare di caricarli
model_path_keras = MODEL_FILENAME
scaler_path_joblib = SCALER_FILENAME

# Controlli di esistenza dei file per un feedback migliore all'utente
if not os.path.exists(model_path_keras):
    st.error(f"File del modello Keras '{model_path_keras}' non trovato! Assicurati che sia nella stessa cartella di app.py.")
if not os.path.exists(scaler_path_joblib):
    st.error(f"File dello scaler '{scaler_path_joblib}' non trovato! Assicurati che sia nella stessa cartella di app.py.")

# Solo se i file esistono, prova a caricare i modelli
sbert_predictor = None
scaler_predictor = None
keras_predictor = None

# Tentativo di caricamento dei modelli solo se i file esistono
if os.path.exists(model_path_keras) and os.path.exists(scaler_path_joblib):
    sbert_predictor = load_sbert_model()
    scaler_predictor = load_scaler(scaler_path_joblib)
    keras_predictor = load_keras_classifier(model_path_keras)
else:
    st.warning("Impossibile caricare i modelli. Verifica la presenza dei file e riavvia l'app se necessario.")


user_input = st.text_area("Inserisci una frase da analizzare (in inglese):", height=100, placeholder="Es: people like this are...")

if st.button("Analizza Frase"):
    if sbert_predictor and scaler_predictor and keras_predictor: # Controlla che i modelli siano caricati
        if user_input:
            with st.spinner("Analisi in corso..."):
                prediction_label, probability, error_message = predict(user_input, sbert_predictor, scaler_predictor, keras_predictor)

            if error_message:
                st.error(error_message)
            elif prediction_label is not None:
                st.subheader("Risultato dell'Analisi:")
                if prediction_label == "Discriminatoria":
                    st.error(f"🔴 Predizione: {prediction_label}")
                else:
                    st.success(f"🟢 Predizione: {prediction_label}")
                st.write(f"Probabilità che sia discriminatoria: {probability:.2%}")


                prob_text = f"{probability*100:.1f}%"
                if probability > 0.75:
                    color = "red"
                elif probability > 0.5:
                    color = "orange"
                else:
                    color = "green"

                st.markdown(f"""
                <div style="
                    background-color: #f0f2f6;
                    border-radius: 5px;
                    padding: 10px;
                    margin-top:10px;
                ">
                    <div style="font-size: 1em; text-align: center;">Confidenza della predizione 'Discriminatoria'</div>
                    <div style="
                        width: 100%;
                        background-color: #ddd;
                        border-radius: 5px;
                        margin-top: 5px;
                    ">
                        <div style="
                            width: {probability*100}%;
                            background-color: {color};
                            text-align: center;
                            line-height: 20px;
                            color: white;
                            font-size: 0.8em;
                            border-radius: 5px;
                            height: 20px;
                        ">
                            {prob_text}
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            else:
                st.warning("Nessun input fornito o problema sconosciuto.")
        else:
            st.warning("Per favore, inserisci una frase da analizzare.")
    else:
        st.error("I modelli non sono stati caricati correttamente. Impossibile procedere con l'analisi.")

st.markdown("---")
st.markdown("Realizzato con Streamlit. Modello addestrato su dati specifici.")
st.markdown("Sviluppato da Christian Aloise")
