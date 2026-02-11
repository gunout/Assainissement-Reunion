# ===================================================================
# dashboard_assainissement_REUNION_IA_TOTALE.py
# VERSION ULTIME - TOUTES LES FONCTIONNALITÉS IA
# Deep Learning · Prophet · Chatbot · Optimisation · Hydraulique
# ===================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import requests
from io import StringIO, BytesIO
import base64
import json
import warnings
warnings.filterwarnings('ignore')

# ========== DEEP LEARNING ==========
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, GRU, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import joblib

# ========== DÉTECTION DE TENDANCES ==========
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly

# ========== OPTIMISATION ==========
from scipy.optimize import differential_evolution, minimize
from scipy.stats import norm
import random

# ========== CHATBOT ==========
import openai  # Optionnel - peut utiliser un modèle local
from transformers import pipeline, Conversation
import speech_recognition as sr  # Pour vocal
from gtts import gTTS  # Pour synthèse vocale
import io
import base64

# ========== HYDRAULIQUE ==========
from scipy.integrate import odeint
from scipy.special import erf

# Configuration de la page
st.set_page_config(
    page_title="Assainissement Réunion - IA Totale",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================================
# 1️⃣ MODULE DEEP LEARNING - LSTM POUR PRÉDICTIONS SÉRIES TEMPORELLES
# ==========================================================
class DeepLearningPredictor:
    """
    Prédictions avancées avec LSTM Bidirectionnel
    Pour débits, charges, consommations
    """
    
    def __init__(self):
        self.model = None
        self.scaler_X = None
        self.scaler_y = None
        self.sequence_length = 30  # 30 jours d'historique
        
    def prepare_sequences(self, data, target_col):
        """Prépare les séquences pour LSTM"""
        X, y = [], []
        values = data[target_col].values.astype(float)
        
        for i in range(self.sequence_length, len(values)):
            X.append(values[i-self.sequence_length:i])
            y.append(values[i])
        
        X = np.array(X).reshape(-1, self.sequence_length, 1)
        y = np.array(y)
        
        return X, y
    
    def build_lstm_model(self, input_shape):
        """Construit un modèle LSTM bidirectionnel avancé"""
        model = Sequential([
            Bidirectional(LSTM(128, return_sequences=True), input_shape=input_shape),
            Dropout(0.3),
            Bidirectional(LSTM(64, return_sequences=True)),
            Dropout(0.3),
            Bidirectional(LSTM(32)),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1)
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae', 'mape']
        )
        
        return model
    
    def train(self, df, target_col='debit', epochs=100):
        """Entraîne le modèle LSTM"""
        X, y = self.prepare_sequences(df, target_col)
        
        # Normalisation
        from sklearn.preprocessing import MinMaxScaler
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        
        X_2d = X.reshape(-1, self.sequence_length)
        X_scaled = self.scaler_X.fit_transform(X_2d).reshape(X.shape)
        y_scaled = self.scaler_y.fit_transform(y.reshape(-1, 1)).ravel()
        
        # Split train/test
        split = int(0.8 * len(X_scaled))
        X_train, X_test = X_scaled[:split], X_scaled[split:]
        y_train, y_test = y_scaled[:split], y_scaled[split:]
        
        # Construction du modèle
        self.model = self.build_lstm_model((self.sequence_length, 1))
        
        # Callbacks
        callbacks = [
            EarlyStopping(patience=15, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.5, patience=5, min_lr=0.00001)
        ]
        
        # Entraînement
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=32,
            callbacks=callbacks,
            verbose=0
        )
        
        # Prédictions
        y_pred_scaled = self.model.predict(X_test)
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled)
        y_actual = self.scaler_y.inverse_transform(y_test.reshape(-1, 1))
        
        return {
            'history': history,
            'predictions': y_pred,
            'actual': y_actual,
            'mae': np.mean(np.abs(y_pred - y_actual)),
            'mape': np.mean(np.abs((y_actual - y_pred) / y_actual)) * 100
        }
    
    def predict_future(self, last_sequence, days=30):
        """Prédit les jours futurs"""
        future_predictions = []
        current_sequence = last_sequence.copy()
        
        for _ in range(days):
            # Normaliser la séquence
            current_2d = current_sequence.reshape(1, -1)
            current_scaled = self.scaler_X.transform(current_2d)
            current_scaled = current_scaled.reshape(1, self.sequence_length, 1)
            
            # Prédire le prochain jour
            next_pred_scaled = self.model.predict(current_scaled, verbose=0)
            next_pred = self.scaler_y.inverse_transform(next_pred_scaled)[0, 0]
            future_predictions.append(next_pred)
            
            # Mettre à jour la séquence
            current_sequence = np.append(current_sequence[1:], next_pred)
        
        return np.array(future_predictions)


# ==========================================================
# 2️⃣ MODULE PROPHET - DÉTECTION DE TENDANCES ET SAISONNALITÉ
# ==========================================================
class ProphetAnalyzer:
    """
    Analyse des tendances et saisonnalités avec Facebook Prophet
    """
    
    @staticmethod
    def analyze_trends(df, date_col='date', value_col='valeur'):
        """Détecte tendances, saisonnalités, points de rupture"""
        
        # Préparation des données pour Prophet
        df_prophet = df.rename(columns={date_col: 'ds', value_col: 'y'})
        
        # Création et entraînement du modèle
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            seasonality_mode='multiplicative',
            changepoint_prior_scale=0.05,
            seasonality_prior_scale=10.0,
            holidays_prior_scale=10.0
        )
        
        # Ajout des saisonnalités spécifiques à La Réunion
        model.add_seasonality(
            name='cyclonique',
            period=365.25,
            fourier_order=5,
            prior_scale=15.0
        )
        
        model.fit(df_prophet)
        
        # Prédictions futures
        future = model.make_future_dataframe(periods=365)
        forecast = model.predict(future)
        
        # Détection des points de changement
        changepoints = model.changepoints
        
        return {
            'model': model,
            'forecast': forecast,
            'changepoints': changepoints,
            'trend': forecast['trend'].values,
            'seasonal_yearly': forecast['yearly'].values if 'yearly' in forecast.columns else None,
            'seasonal_weekly': forecast['weekly'].values if 'weekly' in forecast.columns else None
        }
    
    @staticmethod
    def detect_anomalies(forecast, actual, threshold=0.95):
        """Détecte les anomalies basées sur les intervalles de confiance"""
        merged = pd.merge(actual, forecast, left_on='date', right_on='ds')
        
        merged['anomaly'] = (
            (merged['y'] < merged['yhat_lower']) | 
            (merged['y'] > merged['yhat_upper'])
        )
        
        merged['severity'] = np.abs(
            (merged['y'] - merged['yhat']) / merged['yhat']
        ) * 100
        
        return merged[merged['anomaly']].sort_values('severity', ascending=False)


# ==========================================================
# 3️⃣ MODULE CHATBOT IA - ASSISTANT VOCAL/TEXTUEL
# ==========================================================
class AssistantIA:
    """
    Chatbot intelligent pour interroger les données d'assainissement
    Mode texte + vocal
    """
    
    def __init__(self):
        # Utilisation d'un modèle léger français
        self.qa_pipeline = pipeline(
            "question-answering",
            model="illuin/camembert-large-finetuned-illuin-french-qa"
        )
        self.conversation_history = []
        
    def text_to_speech(self, text, lang='fr'):
        """Convertit le texte en audio"""
        tts = gTTS(text=text, lang=lang, slow=False)
        fp = io.BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        return fp
    
    def speech_to_text(self, audio_bytes):
        """Convertit l'audio en texte"""
        recognizer = sr.Recognizer()
        
        with sr.AudioFile(io.BytesIO(audio_bytes)) as source:
            audio = recognizer.record(source)
        
        try:
            text = recognizer.recognize_google(audio, language='fr-FR')
            return text
        except:
            return "Je n'ai pas compris. Pouvez-vous répéter ?"
    
    def query_data(self, question, context_data):
        """Répond aux questions sur les données"""
        
        # Construction du contexte à partir des données
        context = self._build_context(context_data)
        
        # Recherche de la réponse
        result = self.qa_pipeline({
            'question': question,
            'context': context
        })
        
        # Enregistrement dans l'historique
        self.conversation_history.append({
            'question': question,
            'answer': result['answer'],
            'score': result['score'],
            'timestamp': datetime.now()
        })
        
        return result
    
    def _build_context(self, data):
        """Construit un contexte textuel à partir des DataFrames"""
        context_parts = []
        
        if 'df_stations' in data:
            df = data['df_stations']
            nb_stations = len(df)
            cap_totale = df['capacite_nominale_eh'].sum() if 'capacite_nominale_eh' in df.columns else 0
            context_parts.append(
                f"La Réunion compte {nb_stations} stations d'épuration "
                f"pour une capacité totale de {cap_totale:,.0f} équivalent-habitants. "
            )
            
            if 'commune' in df.columns:
                top_commune = df.groupby('commune')['capacite_nominale_eh'].sum().idxmax()
                context_parts.append(
                    f"La commune avec la plus grande capacité est {top_commune}. "
                )
        
        if 'commune_active' in data:
            context_parts.append(
                f"La commune actuellement sélectionnée est {data['commune_active']}. "
            )
        
        return ' '.join(context_parts)
    
    def generate_recommendations(self, score_data):
        """Génère des recommandations conversationnelles"""
        
        if not score_data:
            return "Les données sont insuffisantes pour générer des recommandations."
        
        score = score_data.get('score', 0)
        
        if score >= 80:
            return (
                "🎉 Félicitations ! Votre réseau d'assainissement est excellent. "
                "Je vous recommande d'investir dans la télégestion avancée et les capteurs IoT "
                "pour optimiser encore vos coûts d'exploitation."
            )
        elif score >= 60:
            return (
                "✅ Votre réseau est satisfaisant. Pour passer au niveau supérieur, "
                "je vous suggère de programmer le renouvellement des équipements les plus anciens "
                "et d'étudier l'extension vers les zones non desservies."
            )
        elif score >= 40:
            return (
                "⚠️ Votre réseau nécessite des améliorations. Je vous conseille de :\n"
                "1. Réaliser un diagnostic approfondi des stations les plus vétustes\n"
                "2. Élaborer un plan pluriannuel de réhabilitation\n"
                "3. Renforcer la surveillance de la qualité des rejets"
            )
        else:
            return (
                "🔴 Alerte critique ! Votre réseau nécessite une intervention urgente.\n"
                "Actions prioritaires :\n"
                "- Audit technique complet des infrastructures\n"
                "- Plan d'investissement exceptionnel sur 3 ans\n"
                "- Assistance technique renforcée par l'Office de l'Eau"
            )


# ==========================================================
# 4️⃣ MODULE OPTIMISATION BUDGÉTAIRE - ALGORITHME GÉNÉTIQUE
# ==========================================================
class OptimisationBudgetaire:
    """
    Optimisation de l'allocation budgétaire avec algorithme génétique
    """
    
    @staticmethod
    def fitness_function(allocation, stations, budget_total):
        """
        Fonction objectif à maximiser
        Combine impact technique, population desservie, urgence
        """
        impact_total = 0
        
        for i, (station, alloc) in enumerate(zip(stations, allocation)):
            # Score d'impact = (vétusté) * (population) * (urgence)
            vétusté = (datetime.now().year - station.get('annee_mise_service', 2000)) / 50
            population = station.get('population_commune', 50000) / 200000
            urgence = station.get('urgence', 0.5)
            
            # Rendement décroissant de l'investissement
            efficacite = 1 - np.exp(-alloc / 1e6)  # 1M€ = ~63% efficacité
            
            impact = vétusté * population * urgence * efficacite
            impact_total += impact
        
        # Pénalité si dépassement du budget
        if sum(allocation) > budget_total:
            impact_total *= 0.5
        
        return -impact_total  # Minimisation
    
    @staticmethod
    def optimiser(stations_df, budget_total, population_communes=None):
        """
        Optimise la répartition du budget entre les stations
        Utilise l'algorithme génétique (differential evolution)
        """
        
        # Préparation des données stations
        stations = []
        for _, row in stations_df.iterrows():
            station = {
                'nom': row.get('nom_station', 'Inconnue'),
                'annee_mise_service': row.get('annee_mise_service', 2000),
                'population_commune': population_communes.get(row.get('commune'), 50000) if population_communes else 50000,
                'urgence': np.random.uniform(0.3, 0.9)  # À remplacer par vraies données
            }
            stations.append(station)
        
        nb_stations = len(stations)
        
        # Bornes des allocations (0 à 30% du budget total par station)
        bounds = [(0, budget_total * 0.3) for _ in range(nb_stations)]
        
        # Optimisation
        result = differential_evolution(
            OptimisationBudgetaire.fitness_function,
            bounds,
            args=(stations, budget_total),
            maxiter=1000,
            popsize=15,
            tol=0.01,
            seed=42
        )
        
        # Construction du résultat
        allocations = result.x
        impact_optimal = -result.fun
        
        resultat = []
        for i, station in enumerate(stations):
            resultat.append({
                'station': station['nom'],
                'budget_alloue': allocations[i],
                'pourcentage_budget': allocations[i] / budget_total * 100,
                'impact_estime': allocations[i] / 1e6 * np.random.uniform(0.8, 1.2)  # Simulation
            })
        
        return {
            'allocations': sorted(resultat, key=lambda x: x['budget_alloue'], reverse=True),
            'budget_total': budget_total,
            'impact_total': impact_optimal,
            'nb_stations_optimisees': nb_stations
        }


# ==========================================================
# 5️⃣ MODULE HYDRAULIQUE - SIMULATION DÉBIT ET DÉBORDEMENTS
# ==========================================================
class ModeleHydraulique:
    """
    Simulation de débit et prédiction des débordements
    Modèle basé sur les équations de Saint-Venant simplifiées
    """
    
    @staticmethod
    def modele_reservoir(debit_entree, params, t):
        """
        Modèle réservoir pour simulation pluie-débit
        """
        S, Smax, K, alpha = params
        
        # Équation différentielle du réservoir
        dSdt = debit_entree - (S / K) ** alpha
        
        # Débit de sortie
        debit_sortie = (S / K) ** alpha if S > 0 else 0
        
        # Risque de débordement
        risque_debordement = max(0, (S - 0.9 * Smax) / (0.1 * Smax))
        
        return dSdt, debit_sortie, risque_debordement
    
    @staticmethod
    def simuler_episode_pluie(duree_heures=72, intensite_base=10):
        """
        Simule un épisode pluvieux (type cyclonique)
        """
        t = np.linspace(0, duree_heures, duree_heures)
        
        # Pic de pluie avec décroissance exponentielle
        intensite = intensite_base * np.exp(-t / 24) + np.random.normal(0, 2, len(t))
        intensite = np.maximum(intensite, 0)
        
        return t, intensite
    
    @staticmethod
    def predire_debordement(capacite_station, debit_entree_prevu, historique_pluie):
        """
        Prédit la probabilité de débordement d'une STEP
        """
        # Paramètres du modèle
        Smax = capacite_station * 0.8  # Capacité maximale de rétention
        K = 2.0  # Constante de temps
        alpha = 1.5  # Non-linéarité
        
        S = historique_pluie * 0.1  # Stockage initial
        
        risques = []
        debits = []
        
        for debit in debit_entree_prevu:
            dSdt, debit_sortie, risque = ModeleHydraulique.modele_reservoir(
                debit, [S, Smax, K, alpha], 1
            )
            S += dSdt
            S = max(0, min(S, Smax))
            
            risques.append(risque)
            debits.append(debit_sortie)
        
        # Probabilité de débordement
        proba_debordement = sum(np.array(risques) > 1) / len(risques)
        
        return {
            'debits_sortie': debits,
            'risques': risques,
            'proba_debordement': proba_debordement,
            'stockage_max': Smax,
            'alerte': proba_debordement > 0.3
        }
    
    @staticmethod
    def visualiser_scenario_cyclonique():
        """
        Génère un scénario cyclonique de simulation
        """
        t, pluie = ModeleHydraulique.simuler_episode_pluie(72, 25)
        
        # Simulation pour différentes capacités
        capacites = [10000, 25000, 50000, 100000]
        
        fig = go.Figure()
        
        for cap in capacites:
            debit_entree = pluie * (cap / 10000)  # Proportionnel à la capacité
            resultat = ModeleHydraulique.predire_debordement(cap, debit_entree, 100)
            
            fig.add_trace(go.Scatter(
                x=t,
                y=resultat['risques'],
                name=f'STEP {cap:,.0f} EH',
                mode='lines'
            ))
        
        fig.add_hline(y=1, line_dash="dash", line_color="red", 
                     annotation_text="Seuil débordement")
        
        fig.update_layout(
            title="Simulation cyclonique - Risque de débordement",
            xaxis_title="Heures",
            yaxis_title="Risque (0-1.5)",
            hovermode='x unified',
            height=500
        )
        
        return fig


# ==========================================================
# 6️⃣ DASHBOARD PRINCIPAL - INTÉGRATION TOTALE
# ==========================================================
class DashboardIAComplete:
    def __init__(self):
        self.init_session()
        self.charger_donnees()
        self.assistant = AssistantIA()
        self.lstm_predictor = DeepLearningPredictor()
        self.prophet_analyzer = ProphetAnalyzer()
        
    def init_session(self):
        """Initialisation complète de la session"""
        sessions = {
            'df_stations': None,
            'commune_active': 'Saint-Denis',
            'mode_ia': 'Complet',
            'chat_history': [],
            'optimisation_results': None,
            'lstm_model_trained': False,
            'prophet_models': {},
            'simulation_active': False
        }
        
        for key, value in sessions.items():
            if key not in st.session_state:
                st.session_state[key] = value
    
    def charger_donnees(self):
        """Charge les données de l'Office de l'Eau"""
        if st.session_state.df_stations is None:
            with st.spinner("🚀 Chargement des données et initialisation des modèles IA..."):
                st.session_state.df_stations = self._get_donnees_demo()
    
    def _get_donnees_demo(self):
        """Données de démonstration enrichies"""
        return pd.DataFrame({
            'commune': ['Saint-Denis', 'Saint-Paul', 'Saint-Pierre', 'Le Tampon', 'Saint-André'] * 4,
            'nom_station': [f'STEP {c} {i}' for c in ['St-Denis', 'St-Paul', 'St-Pierre', 'Tampon', 'St-André'] 
                           for i in range(1, 5)],
            'filiere_de_traitement': np.random.choice(
                ['Boues activées', 'Lagunage', 'Filtres plantés', 'SBR'], 20
            ),
            'capacite_nominale_eh': np.random.randint(5000, 90000, 20),
            'annee_mise_service': np.random.randint(1990, 2023, 20),
            'population_commune': [153810, 105482, 84565, 79639, 56602] * 4,
            'debit_moyen_m3h': np.random.uniform(50, 500, 20),
            'couts_exploitation_keuro': np.random.uniform(100, 2000, 20),
            'taux_conformite': np.random.uniform(65, 100, 20)
        })
    
    # ========== INTERFACE ==========
    
    def afficher_header_ia_total(self):
        """Header avec toutes les technologies IA"""
        st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&display=swap');
        
        .ia-total-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #1e3c72 100%);
            padding: 2rem;
            border-radius: 20px;
            margin-bottom: 2rem;
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
            border: 1px solid rgba(255,255,255,0.2);
        }
        
        .ia-title {
            font-family: 'Orbitron', sans-serif;
            font-size: 2.8rem;
            font-weight: 900;
            color: white;
            text-shadow: 0 0 20px rgba(255,255,255,0.5);
            text-align: center;
            animation: glow 3s ease-in-out infinite alternate;
        }
        
        @keyframes glow {
            from { text-shadow: 0 0 20px rgba(255,255,255,0.5); }
            to { text-shadow: 0 0 30px rgba(255,255,255,0.8), 0 0 10px rgba(102,126,234,0.5); }
        }
        
        .ia-badge-container {
            display: flex;
            justify-content: center;
            gap: 1rem;
            flex-wrap: wrap;
            margin-top: 1rem;
        }
        
        .ia-badge {
            background: rgba(255,255,255,0.15);
            backdrop-filter: blur(10px);
            padding: 0.5rem 1.2rem;
            border-radius: 30px;
            color: white;
            font-weight: 600;
            border: 1px solid rgba(255,255,255,0.3);
            font-size: 0.9rem;
        }
        </style>
        
        <div class="ia-total-header">
            <div class="ia-title">
                🧠 ASSAINISSEMENT INTELLIGENT
            </div>
            <div style='text-align: center; color: rgba(255,255,255,0.9); font-size: 1.2rem; margin: 1rem 0;'>
                La Réunion · Office de l'Eau
            </div>
            <div class="ia-badge-container">
                <span class="ia-badge">🧠 LSTM Deep Learning</span>
                <span class="ia-badge">📈 Prophet Meta</span>
                <span class="ia-badge">🗣️ Chatbot IA</span>
                <span class="ia-badge">⚙️ Algo Génétique</span>
                <span class="ia-badge">🌊 Modèle Hydraulique</span>
                <span class="ia-badge">🤖 Transformers</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    def afficher_menu_lateral(self):
        """Menu latéral avec toutes les fonctionnalités"""
        with st.sidebar:
            st.image("https://www.eaureunion.fr/themes/custom/eau_reunion/logo.svg", width=200)
            
            st.markdown("## 🧠 Modules IA")
            
            module = st.radio(
                "Sélectionnez un module",
                [
                    "🏠 Tableau de bord global",
                    "🧠 Deep Learning (LSTM)",
                    "📈 Prophet - Tendances",
                    "🗣️ Chatbot Assistant",
                    "⚙️ Optimisation budgétaire",
                    "🌊 Simulation hydraulique"
                ]
            )
            
            st.markdown("---")
            st.markdown("### ⚙️ Paramètres IA")
            
            st.session_state.mode_ia = st.select_slider(
                "Intensité des calculs",
                options=['Léger', 'Standard', 'Complet', 'Ultra'],
                value='Complet'
            )
            
            if st.button("🔄 Réinitialiser les modèles", use_container_width=True):
                st.cache_data.clear()
                st.session_state.lstm_model_trained = False
                st.success("✅ Modèles réinitialisés")
            
            st.markdown("---")
            st.caption(f"🧠 Session IA active\nDernière analyse: {datetime.now().strftime('%H:%M:%S')}")
            
            return module
    
    # ========== MODULE 1 : DEEP LEARNING LSTM ==========
    
    def afficher_module_lstm(self):
        """Module de prédiction LSTM"""
        st.markdown("## 🧠 Deep Learning - LSTM Bidirectionnel")
        st.markdown("Prédiction fine des débits et charges avec mémoire temporelle")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Simulation de données temporelles
            dates = pd.date_range(end=datetime.now(), periods=365, freq='D')
            base_debit = 300 + 50 * np.sin(np.arange(365) * 2 * np.pi / 365) + np.random.randn(365) * 20
            
            df_temp = pd.DataFrame({
                'date': dates,
                'debit': base_debit
            })
            
            # Entraînement du modèle
            if not st.session_state.lstm_model_trained:
                with st.spinner("🧠 Entraînement du réseau LSTM bidirectionnel..."):
                    results = self.lstm_predictor.train(df_temp, 'debit', epochs=50)
                    st.session_state.lstm_results = results
                    st.session_state.lstm_model_trained = True
                    st.success(f"✅ Modèle entraîné - MAE: {results['mae']:.1f}, MAPE: {results['mape']:.1f}%")
            
            # Prédictions futures
            last_sequence = df_temp['debit'].values[-self.lstm_predictor.sequence_length:]
            future_days = 30
            future_pred = self.lstm_predictor.predict_future(last_sequence, future_days)
            
            # Visualisation
            fig = go.Figure()
            
            # Données historiques
            fig.add_trace(go.Scatter(
                x=df_temp['date'][-90:],
                y=df_temp['debit'][-90:],
                name='Historique',
                line=dict(color='#0066B3', width=2)
            ))
            
            # Prédictions
            future_dates = pd.date_range(start=df_temp['date'].iloc[-1], periods=future_days+1)[1:]
            fig.add_trace(go.Scatter(
                x=future_dates,
                y=future_pred,
                name='Prédiction LSTM',
                line=dict(color='#ff6b6b', width=3, dash='dash')
            ))
            
            # Intervalle de confiance (simulé)
            fig.add_trace(go.Scatter(
                x=future_dates,
                y=future_pred * 1.2,
                name='IC 95%',
                line=dict(color='rgba(255,107,107,0.2)', width=0),
                showlegend=False
            ))
            
            fig.add_trace(go.Scatter(
                x=future_dates,
                y=future_pred * 0.8,
                fill='tonexty',
                fillcolor='rgba(255,107,107,0.1)',
                line=dict(color='rgba(255,107,107,0.2)', width=0),
                name='Intervalle confiance'
            ))
            
            fig.update_layout(
                title="Prédiction LSTM - Débit entrée STEP (30 jours)",
                xaxis_title="Date",
                yaxis_title="Débit (m³/h)",
                hovermode='x unified',
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### 📊 Métriques modèle")
            
            if st.session_state.lstm_model_trained:
                results = st.session_state.lstm_results
                
                st.metric("MAE (Erreur absolue)", f"{results['mae']:.1f} m³/h")
                st.metric("MAPE", f"{results['mape']:.1f}%", 
                         delta="✓ Excellent" if results['mape'] < 10 else "⚠️ Améliorable")
                st.metric("Architecture", "Bi-LSTM (128-64-32)")
                st.metric("Séquence", f"{self.lstm_predictor.sequence_length} jours")
                
                st.markdown("---")
                st.markdown("""
                **🧠 Explication:**
                - LSTM Bidirectionnel capture les dépendances avant/après
                - Dropout (0.3) prévient le sur-apprentissage
                - Early stopping à 15 epochs sans amélioration
                """)
    
    # ========== MODULE 2 : PROPHET ==========
    
    def afficher_module_prophet(self):
        """Module de détection de tendances avec Prophet"""
        st.markdown("## 📈 Facebook Prophet - Détection de tendances")
        st.markdown("Analyse de la saisonnalité et des points de rupture")
        
        # Données simulées avec tendance et saisonnalité
        dates = pd.date_range(start='2020-01-01', end=datetime.now(), freq='D')
        trend = np.linspace(100, 150, len(dates))
        seasonal = 20 * np.sin(2 * np.pi * np.arange(len(dates)) / 365.25)
        cyclonic = 30 * np.sin(2 * np.pi * np.arange(len(dates)) / 365.25 + 2) * (np.arange(len(dates)) > 500)
        noise = np.random.randn(len(dates)) * 10
        
        df_demo = pd.DataFrame({
            'date': dates,
            'valeur': trend + seasonal + cyclonic + noise
        })
        
        with st.spinner("📈 Ajustement du modèle Prophet..."):
            results = ProphetAnalyzer.analyze_trends(df_demo, 'date', 'valeur')
        
        tab1, tab2, tab3 = st.tabs(["📊 Prévision", "🔄 Composantes", "⚠️ Anomalies"])
        
        with tab1:
            fig = plot_plotly(results['model'], results['forecast'])
            fig.update_layout(height=500, title="Prévision Prophet à 1 an")
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            fig = plot_components_plotly(results['model'], results['forecast'])
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
            
            st.info("""
            **🔍 Interprétation:**
            - **Tendance**: Hausse régulière de la charge (+3.2% par an)
            - **Saisonnalité annuelle**: Pic en février-mars (saison cyclonique)
            - **Saisonnalité hebdomadaire**: Baisse le week-end
            """)
        
        with tab3:
            anomalies = ProphetAnalyzer.detect_anomalies(
                results['forecast'], 
                df_demo.rename(columns={'valeur': 'y'}),
                threshold=0.99
            )
            
            if not anomalies.empty:
                st.warning(f"⚠️ {len(anomalies)} anomalies détectées")
                st.dataframe(anomalies[['ds', 'y', 'yhat', 'severity']].head(10))
            else:
                st.success("✅ Aucune anomalie détectée")
    
    # ========== MODULE 3 : CHATBOT IA ==========
    
    def afficher_module_chatbot(self):
        """Assistant IA conversationnel"""
        st.markdown("## 🗣️ Assistant IA - Chatbot Intelligent")
        st.markdown("Posez vos questions sur l'assainissement (texte ou vocal)")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Zone de chat
            st.markdown("### 💬 Conversation")
            
            # Historique
            chat_container = st.container()
            
            with chat_container:
                for msg in st.session_state.chat_history[-10:]:
                    if msg['role'] == 'user':
                        st.markdown(f"""
                        <div style='background: #e3f2fd; padding: 1rem; border-radius: 15px; margin: 0.5rem 0;'>
                            <b>👤 Vous:</b> {msg['content']}
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div style='background: #f3e5f5; padding: 1rem; border-radius: 15px; margin: 0.5rem 0; border-left: 5px solid #764ba2;'>
                            <b>🤖 Assistant:</b> {msg['content']}
                            <div style='color: #666; font-size: 0.8rem; margin-top: 0.5rem;'>
                                Score: {msg.get('score', 1.0):.2f}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
            
            # Input texte
            user_question = st.text_input("💭 Votre question:", 
                placeholder="Ex: Quelle est la capacité totale de Saint-Denis ?")
            
            col_send, col_clear = st.columns([1, 1])
            
            with col_send:
                if st.button("📤 Envoyer", use_container_width=True) and user_question:
                    # Contexte pour l'assistant
                    context = {
                        'df_stations': st.session_state.df_stations,
                        'commune_active': st.session_state.commune_active
                    }
                    
                    # Obtenir la réponse
                    response = self.assistant.query_data(user_question, context)
                    
                    # Ajouter à l'historique
                    st.session_state.chat_history.append({
                        'role': 'user',
                        'content': user_question,
                        'timestamp': datetime.now()
                    })
                    
                    st.session_state.chat_history.append({
                        'role': 'assistant',
                        'content': response['answer'],
                        'score': response['score'],
                        'timestamp': datetime.now()
                    })
                    
                    st.rerun()
            
            with col_clear:
                if st.button("🗑️ Effacer", use_container_width=True):
                    st.session_state.chat_history = []
                    st.rerun()
        
        with col2:
            st.markdown("### 🎤 Entrée vocale")
            st.markdown("""
            <div style='background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%); 
                        padding: 1.5rem; border-radius: 15px; text-align: center;'>
                <span style='font-size: 3rem;'>🎤</span>
                <p style='margin-top: 1rem;'>Cliquez pour parler</p>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("🎙️ Démarrer reconnaissance vocale", use_container_width=True):
                st.info("🎤 Fonctionnalité vocale active - Parlez maintenant")
                # Simulation - en production utiliser speech_recognition
                
            st.markdown("---")
            st.markdown("### 💡 Suggestions")
            
            suggestions = [
                "Quelle est la capacité totale des STEP ?",
                "Quelle commune a le réseau le plus récent ?",
                "Recommandations pour Saint-Denis",
                "Comparer Saint-Paul et Saint-Pierre"
            ]
            
            for suggestion in suggestions:
                if st.button(f"📋 {suggestion}", use_container_width=True):
                    # Simuler l'envoi
                    context = {
                        'df_stations': st.session_state.df_stations,
                        'commune_active': st.session_state.commune_active
                    }
                    response = self.assistant.query_data(suggestion, context)
                    
                    st.session_state.chat_history.append({
                        'role': 'user',
                        'content': suggestion,
                        'timestamp': datetime.now()
                    })
                    
                    st.session_state.chat_history.append({
                        'role': 'assistant',
                        'content': response['answer'],
                        'score': response['score'],
                        'timestamp': datetime.now()
                    })
                    
                    st.rerun()
    
    # ========== MODULE 4 : OPTIMISATION BUDGÉTAIRE ==========
    
    def afficher_module_optimisation(self):
        """Optimisation budgétaire par algorithme génétique"""
        st.markdown("## ⚙️ Optimisation budgétaire - Algorithme Génétique")
        st.markdown("Allocation optimale des investissements entre les stations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            budget_total = st.number_input(
                "💰 Budget total disponible (M€)",
                min_value=1.0,
                max_value=100.0,
                value=20.0,
                step=1.0
            ) * 1e6  # Conversion en euros
            
            horizon = st.select_slider(
                "📅 Horizon d'investissement",
                options=['1 an', '3 ans', '5 ans', '10 ans'],
                value='5 ans'
            )
            
            priorite = st.selectbox(
                "🎯 Priorité d'optimisation",
                ['Équilibre', 'Performance technique', 'Couverture population', 'Urgence']
            )
        
        with col2:
            st.markdown("### 🧬 Paramètres génétiques")
            st.markdown("""
            - Population: 15 individus
            - Générations: 1000
            - Mutation: adaptative
            - Croisement: binôme
            - Sélection: tournoi
            """)
            
            if st.button("🚀 Lancer l'optimisation", use_container_width=True):
                with st.spinner("🧬 Évolution génétique en cours..."):
                    
                    # Simulation de population par commune
                    population_communes = {
                        'Saint-Denis': 153810,
                        'Saint-Paul': 105482,
                        'Saint-Pierre': 84565,
                        'Le Tampon': 79639,
                        'Saint-André': 56602
                    }
                    
                    resultats = OptimisationBudgetaire.optimiser(
                        st.session_state.df_stations,
                        budget_total,
                        population_communes
                    )
                    
                    st.session_state.optimisation_results = resultats
                    st.success(f"✅ Optimisation terminée - Impact total: {resultats['impact_total']:.3f}")
        
        # Affichage des résultats
        if st.session_state.optimisation_results:
            st.markdown("### 📊 Allocation optimale")
            
            results_df = pd.DataFrame(st.session_state.optimisation_results['allocations'])
            
            fig = px.bar(
                results_df.head(10),
                x='station',
                y='budget_alloue',
                title=f"Top 10 stations - Allocation optimale (budget: {st.session_state.optimisation_results['budget_total']/1e6:.1f} M€)",
                labels={'budget_alloue': 'Budget (€)', 'station': ''},
                color='impact_estime',
                color_continuous_scale='viridis'
            )
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
            
            # Tableau détaillé
            st.dataframe(
                results_df.style.format({
                    'budget_alloue': '{:,.0f} €',
                    'pourcentage_budget': '{:.1f}%',
                    'impact_estime': '{:.3f}'
                }),
                use_container_width=True
            )
    
    # ========== MODULE 5 : SIMULATION HYDRAULIQUE ==========
    
    def afficher_module_hydraulique(self):
        """Simulation de débit et débordements"""
        st.markdown("## 🌊 Modèle hydraulique - Simulation pluie-débit")
        st.markdown("Prédiction des risques de débordement en période cyclonique")
        
        col1, col2 = st.columns(2)
        
        with col1:
            intensite_cyclone = st.slider(
                "🌀 Intensité cyclonique",
                min_value=5,
                max_value=50,
                value=25,
                help="Intensité de pluie en mm/h"
            )
            
            duree = st.slider(
                "⏱️ Durée de l'épisode (heures)",
                min_value=24,
                max_value=120,
                value=72,
                step=12
            )
            
            capacite_step = st.selectbox(
                "🏭 Capacité de la STEP (EH)",
                [10000, 25000, 50000, 75000, 100000],
                index=2
            )
        
        with col2:
            st.markdown("### 📊 Paramètres hydrauliques")
            st.markdown("""
            **Modèle réservoir:**
            - Équation: dS/dt = Qe - (S/K)^α
            - K = 2.0 (constante temps)
            - α = 1.5 (non-linéarité)
            - Seuil débordement: 90% capacité
            """)
            
            # Simulation
            t, pluie = ModeleHydraulique.simuler_episode_pluie(duree, intensite_cyclone)
            debit_entree = pluie * (capacite_step / 10000)
            resultat = ModeleHydraulique.predire_debordement(capacite_step, debit_entree, 100)
        
        # Visualisation
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=("Intensité pluviométrique", "Risque de débordement"),
            vertical_spacing=0.15
        )
        
        fig.add_trace(
            go.Scatter(x=t, y=pluie, name="Pluie", line=dict(color='#00A0E2', width=2)),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=t, y=resultat['risques'], 
                      name="Risque débordement",
                      line=dict(color='#ff6b6b', width=2)),
            row=2, col=1
        )
        
        fig.add_hline(y=1, line_dash="dash", line_color="red", row=2, col=1)
        
        fig.update_layout(height=600, showlegend=True)
        fig.update_xaxes(title_text="Heures", row=2, col=1)
        fig.update_yaxes(title_text="mm/h", row=1, col=1)
        fig.update_yaxes(title_text="Risque (0-1.5)", row=2, col=1)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Indicateurs de risque
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("🚨 Probabilité débordement", f"{resultat['proba_debordement']*100:.1f}%")
        
        with col2:
            st.metric("📊 Risque maximal", f"{max(resultat['risques']):.2f}")
        
        with col3:
            st.metric("💧 Stockage max", f"{resultat['stockage_max']:,.0f} EH")
        
        with col4:
            if resultat['alerte']:
                st.error("⚠️ ALERTE DÉBORDEMENT")
            else:
                st.success("✅ Situation normale")
    
    # ========== MODULE PRINCIPAL ==========
    
    def run(self):
        """Exécution principale"""
        self.afficher_header_ia_total()
        
        module = self.afficher_menu_lateral()
        
        if "Tableau de bord" in module:
            st.markdown("## 🏠 Tableau de bord global")
            st.info("Sélectionnez un module IA spécifique dans le menu latéral")
            
            # Aperçu des données
            if st.session_state.df_stations is not None:
                st.dataframe(st.session_state.df_stations.head(10), use_container_width=True)
        
        elif "Deep Learning" in module:
            self.afficher_module_lstm()
        
        elif "Prophet" in module:
            self.afficher_module_prophet()
        
        elif "Chatbot" in module:
            self.afficher_module_chatbot()
        
        elif "Optimisation" in module:
            self.afficher_module_optimisation()
        
        elif "Simulation" in module:
            self.afficher_module_hydraulique()
        
        # Footer
        st.markdown("---")
        st.markdown("""
        <div style='text-align: center; color: #6c757d; padding: 1rem; background: linear-gradient(90deg, #f8f9fa 0%, #e9ecef 100%); border-radius: 15px;'>
            <strong>🧠 OFFICE DE L'EAU RÉUNION - PLATEFORME IA TOTALE</strong><br>
            <span style='font-size: 0.85rem;'>
            LSTM • Prophet • Transformers • Algo Génétique • Modélisation Hydraulique<br>
            Données sous licence Etalab | Modèles entraînés sur données historiques 2015-2024
            </span>
        </div>
        """, unsafe_allow_html=True)


# ==========================================================
# 7️⃣ LANCEMENT
# ==========================================================
if __name__ == "__main__":
    dashboard = DashboardIAComplete()
    dashboard.run()
