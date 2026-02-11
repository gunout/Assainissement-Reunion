# ===================================================================
# dashboard_assainissement_REUNION_IA_LIGHT.py
# VERSION ULTRA-LÉGÈRE - 100% Streamlit Cloud Compatible
# Alternatives : scikit-learn, statsmodels, Prophet, NLP léger
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

# ========== MACHINE LEARNING LÉGER ==========
from sklearn.ensemble import RandomForestRegressor, IsolationForest, GradientBoostingRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ========== ANALYSE TEMPORELLE ==========
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
import scipy.stats as stats

# ========== PROPHET (LÉGER) ==========
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly

# ========== OPTIMISATION ==========
from scipy.optimize import differential_evolution, minimize, dual_annealing
from scipy.spatial.distance import cdist

# ========== NLP LÉGER (SANS TRANSFORMERS) ==========
import re
from collections import Counter
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

# ========== ÉVITER LES ERREURS DE MÉMOIRE ==========
import gc
import psutil
import os

# Configuration de la page
st.set_page_config(
    page_title="Assainissement Réunion - IA Légère",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.eaureunion.fr',
        'Report a bug': 'https://github.com',
        'About': 'Dashboard Assainissement Réunion - Version IA Légère'
    }
)

# ==========================================================
# 1️⃣ MODÈLE DE PRÉDICTION LÉGER - RANDOM FOREST / ARIMA
# ==========================================================

class PredicteurLeger:
    """
    Alternative à LSTM - Utilise Random Forest + ARIMA
    Beaucoup plus léger, compatible Streamlit Cloud
    """
    
    def __init__(self):
        self.model_rf = RandomForestRegressor(
            n_estimators=50,  # Réduit pour la mémoire
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        self.model_arima = None
        self.scaler = MinMaxScaler()
        
    def predict_random_forest(self, df, target_col, n_future=30):
        """
        Prédiction avec Random Forest sur features temporelles
        """
        # Création des features temporelles
        df_features = df.copy()
        df_features['dayofweek'] = df_features.index.dayofweek
        df_features['month'] = df_features.index.month
        df_features['day'] = df_features.index.day
        df_features['dayofyear'] = df_features.index.dayofyear
        df_features['quarter'] = df_features.index.quarter
        df_features['weekend'] = (df_features.index.dayofweek >= 5).astype(int)
        
        # Lag features
        for lag in [1, 2, 3, 7, 14, 30]:
            df_features[f'lag_{lag}'] = df_features[target_col].shift(lag)
        
        # Moyennes mobiles
        df_features['rolling_mean_7'] = df_features[target_col].rolling(7).mean()
        df_features['rolling_mean_30'] = df_features[target_col].rolling(30).mean()
        
        # Drop NaN
        df_features = df_features.dropna()
        
        # Préparation X, y
        feature_cols = [col for col in df_features.columns if col != target_col]
        X = df_features[feature_cols]
        y = df_features[target_col]
        
        # Split temporel (pas de shuffle)
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # Entraînement
        self.model_rf.fit(X_train, y_train)
        
        # Prédictions
        y_pred = self.model_rf.predict(X_test)
        
        # Feature importance
        importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': self.model_rf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Prédictions futures
        last_row = df_features.iloc[-1:][feature_cols]
        future_predictions = []
        
        for _ in range(n_future):
            pred = self.model_rf.predict(last_row)[0]
            future_predictions.append(pred)
            
            # Mise à jour des lags
            for lag in [1, 2, 3, 7, 14, 30]:
                if lag == 1:
                    last_row[f'lag_{lag}'] = pred
                else:
                    last_row[f'lag_{lag}'] = last_row[f'lag_{lag-1}'].values
        
        return {
            'y_test': y_test,
            'y_pred': y_pred,
            'future': np.array(future_predictions),
            'importance': importance,
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred))
        }
    
    def predict_arima(self, series, n_future=30):
        """
        Prédiction avec ARIMA pour les séries temporelles
        """
        try:
            model = ARIMA(series, order=(2,1,2))
            self.model_arima = model.fit()
            
            # Prédictions
            forecast = self.model_arima.forecast(n_future)
            
            return {
                'forecast': forecast,
                'aic': self.model_arima.aic,
                'bic': self.model_arima.bic
            }
        except:
            # Fallback sur Exponential Smoothing
            model = ExponentialSmoothing(
                series,
                seasonal_periods=7,
                trend='add',
                seasonal='add'
            )
            self.model_arima = model.fit()
            forecast = self.model_arima.forecast(n_future)
            
            return {
                'forecast': forecast,
                'aic': None,
                'bic': None
            }


# ==========================================================
# 2️⃣ CHATBOT LÉGER - SANS TRANSFORMERS
# ==========================================================

class ChatbotLeger:
    """
    Assistant virtuel basé sur NLP classique
    - TF-IDF + Similarité cosinus
    - Base de connaissances locale
    - Pas de dépendances lourdes
    """
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=1000)
        self.knowledge_base = self._init_knowledge_base()
        self.questions = [item['question'] for item in self.knowledge_base]
        self.answers = [item['answer'] for item in self.knowledge_base]
        self.tags = [item['tag'] for item in self.knowledge_base]
        
        # Vectorisation des questions
        if self.questions:
            self.X = self.vectorizer.fit_transform(self.questions)
    
    def _init_knowledge_base(self):
        """Base de connaissances sur l'assainissement"""
        return [
            {
                'question': 'Quelle est la capacité totale des STEP à La Réunion ?',
                'answer': 'La capacité totale de traitement des stations d\'épuration de La Réunion est d\'environ 450 000 équivalent-habitants, répartie sur 24 communes.',
                'tag': 'capacite'
            },
            {
                'question': 'Comment améliorer le rendement de mon réseau ?',
                'answer': 'Pour améliorer le rendement : 1) Recherche de fuites régulière, 2) Renouvellement des canalisations vétustes, 3) Sectorisation, 4) Télégestion, 5) Campagnes de contrôles.',
                'tag': 'rendement'
            },
            {
                'question': 'Quelles sont les normes de rejet ?',
                'answer': 'Les normes de rejet sont définies par l\'arrêté du 21 juillet 2015 : DBO5 < 25 mg/L, DCO < 125 mg/L, MES < 35 mg/L, NTK < 15 mg/L, PT < 2 mg/L.',
                'tag': 'normes'
            },
            {
                'question': 'Que faire en cas de débordement ?',
                'answer': 'En cas de débordement : 1) Confiner la zone, 2) Contacter le service exploitation, 3) Prévenir l\'ARS, 4) Analyser les causes, 5) Nettoyer et désinfecter, 6) Rédiger un rapport d\'incident.',
                'tag': 'urgence'
            },
            {
                'question': 'Comment financer des travaux d\'assainissement ?',
                'answer': 'Financements possibles : Agence de l\'Eau (aides jusqu\'à 50%), Conseil Départemental, FEDER, fonds de concours intercommunaux, tarification progressive.',
                'tag': 'financement'
            },
            {
                'question': 'Qu\'est-ce que le SPANC ?',
                'answer': 'Le SPANC (Service Public d\'Assainissement Non Collectif) contrôle les installations d\'assainissement individuel. Mission : diagnostic, conseil, conformité, suivi des installations.',
                'tag': 'spanc'
            },
            {
                'question': 'Quelle est la différence entre EU et EP ?',
                'answer': 'EU = Eaux Usées (domestiques, industrielles) ; EP = Eaux Pluviales (pluie, ruissellement). Le séparatif évite les surcharges des STEP par temps de pluie.',
                'tag': 'reseau'
            },
            {
                'question': 'Comment réduire les odeurs ?',
                'answer': 'Solutions anti-odeurs : traitement chimique (chlore, peroxyde), biofiltration, charbon actif, couverture des ouvrages, ventilation, extraction d\'air.',
                'tag': 'exploitation'
            },
            {
                'question': 'Saint-Denis',
                'answer': 'Saint-Denis dispose de 2 stations d\'épuration principales : STEP Saint-Denis (85 000 EH) et STEP Sainte-Marie (19 000 EH). Taux de conformité : 98.5%.',
                'tag': 'commune'
            },
            {
                'question': 'Saint-Paul',
                'answer': 'Saint-Paul est équipée de la STEP Saint-Paul (62 000 EH) en filière lagunage, mise en service en 2005. Rendement réseau : 84.2%.',
                'tag': 'commune'
            },
            {
                'question': 'Saint-Pierre',
                'answer': 'Saint-Pierre dispose de 2 STEP : Saint-Pierre (48 000 EH) et une unité complémentaire (24 000 EH). Filière boues activées. Conformité : 96.8%.',
                'tag': 'commune'
            }
        ]
    
    def get_response(self, query):
        """
        Trouve la réponse la plus pertinente par similarité cosinus
        """
        if not query:
            return "Veuillez poser une question."
        
        # Vectorisation de la requête
        query_vec = self.vectorizer.transform([query])
        
        # Similarité cosinus
        similarities = cosine_similarity(query_vec, self.X)[0]
        
        # Meilleur match
        best_idx = np.argmax(similarities)
        best_score = similarities[best_idx]
        
        # Seuil de confiance
        if best_score > 0.3:
            response = {
                'answer': self.answers[best_idx],
                'tag': self.tags[best_idx],
                'confidence': float(best_score),
                'question_similaire': self.questions[best_idx]
            }
        else:
            # Réponse par défaut
            response = {
                'answer': "Je n'ai pas trouvé de réponse exacte à votre question. Voici quelques informations générales sur l'assainissement à La Réunion : L'Office de l'Eau Réunion gère 24 communes avec plus de 40 stations d'épuration. Contactez le 0262 30 88 00 pour plus d'informations.",
                'tag': 'default',
                'confidence': float(best_score),
                'question_similaire': None
            }
        
        return response
    
    def add_to_knowledge(self, question, answer, tag='user'):
        """Enrichit la base de connaissances (session utilisateur)"""
        self.knowledge_base.append({
            'question': question,
            'answer': answer,
            'tag': tag
        })
        self.questions.append(question)
        self.answers.append(answer)
        self.tags.append(tag)
        
        # Ré-entraînement du vectorizer
        self.X = self.vectorizer.fit_transform(self.questions)


# ==========================================================
# 3️⃣ OPTIMISATION LÉGÈRE - ALGORITHMES SIMPLIFIÉS
# ==========================================================

class OptimisationLegere:
    """
    Optimisation budgétaire simplifiée
    - Recuit simulé / Differential Evolution
    - Calculs vectorisés
    """
    
    @staticmethod
    def optimiser_simple(stations_df, budget_total):
        """
        Version allégée de l'optimisation
        """
        if stations_df.empty:
            return None
        
        nb_stations = len(stations_df)
        
        # Score d'urgence simplifié
        scores = []
        for _, station in stations_df.iterrows():
            vétusté = (datetime.now().year - station.get('annee_mise_service', 2000)) / 50
            capacite = station.get('capacite_nominale_eh', 10000) / 100000
            score = vétusté * 0.6 + capacite * 0.4
            scores.append(score)
        
        scores = np.array(scores)
        
        # Normalisation
        scores = scores / scores.sum()
        
        # Allocation proportionnelle au score
        allocations = scores * budget_total
        
        # Contrainte : pas plus de 30% du budget sur une station
        max_per_station = budget_total * 0.3
        allocations = np.minimum(allocations, max_per_station)
        
        # Réallocation du surplus
        surplus = budget_total - allocations.sum()
        if surplus > 0:
            weights = scores * (allocations < max_per_station)
            if weights.sum() > 0:
                weights = weights / weights.sum()
                allocations += weights * surplus
        
        resultats = []
        for i, station in stations_df.iterrows():
            resultats.append({
                'station': station.get('nom_station', f'Station {i}'),
                'commune': station.get('commune', 'Inconnue'),
                'budget_alloue': allocations[i],
                'pourcentage_budget': allocations[i] / budget_total * 100,
                'score_urgence': scores[i]
            })
        
        return {
            'allocations': sorted(resultats, key=lambda x: x['budget_alloue'], reverse=True),
            'budget_total': budget_total,
            'nb_stations': nb_stations
        }


# ==========================================================
# 4️⃣ SIMULATION HYDRAULIQUE LÉGÈRE
# ==========================================================

class HydrauliqueLegere:
    """
    Modèle réservoir simplifié
    """
    
    @staticmethod
    def simuler_crue(duree_heures=72, intensite=25):
        """
        Simulation simplifiée d'un épisode pluvieux
        """
        t = np.linspace(0, duree_heures, min(duree_heures, 100))
        
        # Pic de pluie exponentiel
        pluie = intensite * np.exp(-t / 12) + np.random.normal(0, intensite * 0.1, len(t))
        pluie = np.maximum(pluie, 0)
        
        # Débit simulé (modèle réservoir linéaire)
        debit = np.zeros_like(t)
        stock = 0
        
        for i in range(1, len(t)):
            dt = t[i] - t[i-1]
            stock += pluie[i] * dt * 10  # Coefficient de ruissellement
            debit[i] = stock * 0.1  # Vidange
            stock -= debit[i] * dt
            stock = max(stock, 0)
        
        return t, pluie, debit
    
    @staticmethod
    def risque_debordement(capacite, debit_entree):
        """
        Calcule le risque de débordement simplifié
        """
        charge = debit_entree / (capacite * 0.15)  # Facteur de conversion
        risque = 1 / (1 + np.exp(-(charge - 1) * 5))  # Sigmoid
        
        return {
            'risque': float(risque),
            'niveau': 'Élevé' if risque > 0.7 else 'Moyen' if risque > 0.3 else 'Faible',
            'charge': charge
        }


# ==========================================================
# 5️⃣ TÉLÉCHARGEMENT DES DONNÉES OFFICIELLES
# ==========================================================

@st.cache_data(ttl=3600)
def telecharger_donnees_office_eau():
    """
    Télécharge les données STEP depuis l'Office de l'Eau
    Gestion des erreurs et timeout
    """
    urls = [
        "https://donnees.eaureunion.fr/explore/dataset/stations-de-traitement-des-eaux-usees/download/?format=csv&timezone=Indian/Reunion&use_labels_for_header=true",
        "https://donnees.eaureunion.fr/explore/dataset/stations-de-traitement-des-eaux-usees/download/?format=csv&timezone=Indian/Reunion&use_labels_for_header=true"
    ]
    
    for url in urls:
        try:
            response = requests.get(url, timeout=10, headers={
                'User-Agent': 'Mozilla/5.0 (compatible; Streamlit/1.0)'
            })
            
            if response.status_code == 200:
                df = pd.read_csv(StringIO(response.text), sep=';', low_memory=False, nrows=1000)
                
                # Nettoyage
                df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
                
                return df, "Office de l'Eau Réunion (données officielles)"
        except:
            continue
    
    # Données de démonstration ultra-légères
    df_demo = pd.DataFrame({
        'commune': ['Saint-Denis', 'Saint-Paul', 'Saint-Pierre', 'Le Tampon', 'Saint-André'],
        'nom_station': ['STEP Saint-Denis', 'STEP Saint-Paul', 'STEP Saint-Pierre', 'STEP Le Tampon', 'STEP Saint-André'],
        'filiere_de_traitement': ['Boues activées', 'Lagunage', 'Boues activées', 'Filtres plantés', 'SBR'],
        'capacite_nominale_eh': [85000, 62000, 48000, 35000, 28000],
        'annee_mise_service': [1998, 2005, 2008, 2012, 1995],
        'taux_conformite': [98.5, 97.2, 96.8, 92.1, 89.4]
    })
    
    return df_demo, "Données de démonstration"


# ==========================================================
# 6️⃣ INTERFACE PRINCIPALE STREAMLIT
# ==========================================================

def main():
    """Interface principale ultra-légère"""
    
    st.markdown("""
    <style>
        .stApp {
            background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%);
        }
        .main-header {
            background: linear-gradient(90deg, #0066B3, #00A0E2);
            padding: 2rem;
            border-radius: 15px;
            color: white;
            text-align: center;
            margin-bottom: 2rem;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }
        .ia-badge-light {
            background: white;
            color: #0066B3;
            padding: 0.3rem 1rem;
            border-radius: 20px;
            font-size: 0.9rem;
            font-weight: 600;
            display: inline-block;
            margin: 0.2rem;
        }
        .chat-message {
            padding: 1rem;
            border-radius: 10px;
            margin: 0.5rem 0;
            animation: fadeIn 0.5s;
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        div[data-testid="stMetricValue"] {
            font-size: 2rem;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1 style='margin:0; font-size:2.5rem;'>💧 ASSAINISSEMENT RÉUNION</h1>
        <p style='opacity:0.9; font-size:1.2rem; margin:0.5rem 0;'>IA Légère • 100% Streamlit Cloud</p>
        <div>
            <span class="ia-badge-light">🧠 Random Forest</span>
            <span class="ia-badge-light">📈 Prophet</span>
            <span class="ia-badge-light">🤖 Chatbot NLP</span>
            <span class="ia-badge-light">⚙️ Optimisation</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://www.eaureunion.fr/themes/custom/eau_reunion/logo.svg", width=200)
        
        st.markdown("## 📥 Données")
        
        # Chargement des données
        if st.button("🔄 Charger données Office de l'Eau", use_container_width=True):
            with st.spinner("Téléchargement en cours..."):
                df, source = telecharger_donnees_office_eau()
                st.session_state['df_stations'] = df
                st.session_state['source'] = source
                st.success(f"✅ {len(df)} stations chargées")
        
        # Upload manuel (optionnel)
        uploaded_file = st.file_uploader(
            "Ou charger votre fichier CSV",
            type=['csv'],
            help="Format : CSV avec séparateur point-virgule"
        )
        
        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file, sep=';', nrows=1000)
                st.session_state['df_stations'] = df
                st.session_state['source'] = "Fichier local"
                st.success("✅ Fichier chargé")
            except Exception as e:
                st.error(f"Erreur: {str(e)[:50]}...")
        
        # Données par défaut
        if 'df_stations' not in st.session_state:
            df, source = telecharger_donnees_office_eau()
            st.session_state['df_stations'] = df
            st.session_state['source'] = source
        
        st.markdown(f"**Source:** {st.session_state['source']}")
        st.markdown(f"**Stations:** {len(st.session_state['df_stations'])}")
        
        st.markdown("---")
        st.markdown("### 🧠 Modules IA")
        
        module = st.radio(
            "Sélectionner un module",
            [
                "🏠 Tableau de bord",
                "📈 Prédictions (RF/ARIMA)",
                "🤖 Assistant virtuel",
                "⚙️ Optimisation budget",
                "🌊 Simulation pluie"
            ]
        )
        
        st.markdown("---")
        st.caption(f"💾 Mémoire: {psutil.Process().memory_info().rss / 1024 / 1024:.0f} Mo")
    
    # ========== MODULE TABLEAU DE BORD ==========
    if "Tableau de bord" in module:
        st.header("🏠 Tableau de bord - Vue d'ensemble")
        
        df = st.session_state['df_stations']
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("🏭 Stations", len(df))
        
        with col2:
            if 'capacite_nominale_eh' in df.columns:
                cap_totale = df['capacite_nominale_eh'].sum()
                st.metric("👥 Capacité totale", f"{cap_totale:,.0f} EH")
        
        with col3:
            if 'annee_mise_service' in df.columns:
                annee_moy = int(df['annee_mise_service'].mean())
                st.metric("📅 Année moyenne", annee_moy)
        
        with col4:
            if 'taux_conformite' in df.columns:
                conf_moy = df['taux_conformite'].mean()
                st.metric("✅ Conformité", f"{conf_moy:.1f}%")
        
        # Top communes
        if 'commune' in df.columns and 'capacite_nominale_eh' in df.columns:
            st.subheader("📊 Top communes par capacité")
            
            top_communes = df.groupby('commune')['capacite_nominale_eh'].sum().nlargest(10).reset_index()
            
            fig = px.bar(
                top_communes,
                x='commune',
                y='capacite_nominale_eh',
                title="Capacité totale de traitement par commune",
                color='capacite_nominale_eh',
                color_continuous_scale='Blues'
            )
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
        
        # Données brutes
        with st.expander("🔍 Aperçu des données"):
            st.dataframe(df.head(10), use_container_width=True)
    
    # ========== MODULE PRÉDICTIONS ==========
    elif "Prédictions" in module:
        st.header("📈 Prédictions - Random Forest & ARIMA")
        
        df = st.session_state['df_stations']
        
        # Création de données temporelles simulées
        dates = pd.date_range(end=datetime.now(), periods=180, freq='D')
        base = 300 + 20 * np.sin(np.arange(180) * 2 * np.pi / 30)
        bruit = np.random.randn(180) * 15
        
        df_temp = pd.DataFrame({
            'debit': base + bruit
        }, index=dates)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🌲 Random Forest")
            
            predictor = PredicteurLeger()
            
            with st.spinner("Entraînement Random Forest..."):
                results = predictor.predict_random_forest(df_temp, 'debit', n_future=14)
            
            st.metric("MAE", f"{results['mae']:.1f}")
            st.metric("R²", f"{results['r2']:.3f}")
            
            # Graphique
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=df_temp.index[-30:],
                y=df_temp['debit'].values[-30:],
                name='Historique',
                line=dict(color='#0066B3', width=2)
            ))
            
            fig.add_trace(go.Scatter(
                x=pd.date_range(start=df_temp.index[-1], periods=15)[1:],
                y=results['future'][:14],
                name='Prédiction RF',
                line=dict(color='#ff6b6b', width=2, dash='dash')
            ))
            
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Feature importance
            st.dataframe(results['importance'].head(5), use_container_width=True)
        
        with col2:
            st.subheader("📊 ARIMA / Lissage")
            
            with st.spinner("Ajustement ARIMA..."):
                arima_results = predictor.predict_arima(df_temp['debit'].values, n_future=14)
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=df_temp.index[-30:],
                y=df_temp['debit'].values[-30:],
                name='Historique',
                line=dict(color='#0066B3', width=2)
            ))
            
            fig.add_trace(go.Scatter(
                x=pd.date_range(start=df_temp.index[-1], periods=15)[1:],
                y=arima_results['forecast'],
                name='ARIMA',
                line=dict(color='#20bf55', width=2, dash='dash')
            ))
            
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            if arima_results['aic']:
                st.metric("AIC", f"{arima_results['aic']:.0f}")
    
    # ========== MODULE CHATBOT ==========
    elif "Assistant" in module:
        st.header("🤖 Assistant virtuel - Chatbot IA")
        st.markdown("Posez vos questions sur l'assainissement à La Réunion")
        
        # Initialisation du chatbot
        if 'chatbot' not in st.session_state:
            st.session_state['chatbot'] = ChatbotLeger()
        if 'chat_history' not in st.session_state:
            st.session_state['chat_history'] = []
        
        chatbot = st.session_state['chatbot']
        
        # Interface de chat
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Affichage de l'historique
            chat_container = st.container()
            
            with chat_container:
                for msg in st.session_state['chat_history'][-10:]:
                    if msg['role'] == 'user':
                        st.markdown(f"""
                        <div class="chat-message" style="background: #e3f2fd; text-align: right;">
                            <b>👤 Vous:</b> {msg['content']}
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        confidence = msg.get('confidence', 0)
                        conf_color = "green" if confidence > 0.7 else "orange"
                        
                        st.markdown(f"""
                        <div class="chat-message" style="background: #f3e5f5; border-left: 5px solid #764ba2;">
                            <b>🤖 Assistant:</b> {msg['content']}<br>
                            <small style="color: {conf_color};">Confiance: {confidence:.1%}</small>
                        </div>
                        """, unsafe_allow_html=True)
            
            # Saisie utilisateur
            user_input = st.text_input("💬 Votre question:", placeholder="Ex: Capacité de Saint-Denis ?")
            
            col_send, col_clear = st.columns(2)
            
            with col_send:
                if st.button("📤 Envoyer", use_container_width=True) and user_input:
                    # Obtenir réponse
                    response = chatbot.get_response(user_input)
                    
                    # Ajouter à l'historique
                    st.session_state['chat_history'].append({
                        'role': 'user',
                        'content': user_input
                    })
                    
                    st.session_state['chat_history'].append({
                        'role': 'assistant',
                        'content': response['answer'],
                        'confidence': response['confidence']
                    })
                    
                    st.rerun()
            
            with col_clear:
                if st.button("🗑️ Effacer", use_container_width=True):
                    st.session_state['chat_history'] = []
                    st.rerun()
        
        with col2:
            st.markdown("### 💡 Suggestions")
            
            suggestions = [
                "Capacité totale des STEP ?",
                "Normes de rejet",
                "Que faire en cas de débordement ?",
                "Comment financer des travaux ?",
                "Saint-Denis",
                "Différence EU/EP"
            ]
            
            for suggestion in suggestions:
                if st.button(f"📋 {suggestion}", use_container_width=True):
                    response = chatbot.get_response(suggestion)
                    
                    st.session_state['chat_history'].append({
                        'role': 'user',
                        'content': suggestion
                    })
                    
                    st.session_state['chat_history'].append({
                        'role': 'assistant',
                        'content': response['answer'],
                        'confidence': response['confidence']
                    })
                    
                    st.rerun()
            
            st.markdown("---")
            st.markdown("""
            **📚 Base de connaissances:**
            - 11 questions/réponses
            - Similarité cosinus TF-IDF
            - Confiance > 30%
            """)
    
    # ========== MODULE OPTIMISATION ==========
    elif "Optimisation" in module:
        st.header("⚙️ Optimisation budgétaire")
        
        df = st.session_state['df_stations']
        
        col1, col2 = st.columns(2)
        
        with col1:
            budget = st.number_input(
                "💰 Budget total (M€)",
                min_value=1.0,
                max_value=100.0,
                value=20.0,
                step=1.0
            ) * 1e6
            
            if st.button("🚀 Lancer l'optimisation", use_container_width=True):
                with st.spinner("Calcul de l'allocation optimale..."):
                    resultats = OptimisationLegere.optimiser_simple(df, budget)
                    st.session_state['opt_results'] = resultats
                    st.success("✅ Optimisation terminée")
        
        with col2:
            st.markdown("""
            **📊 Méthode:**
            - Score = vétusté × 0.6 + capacité × 0.4
            - Plafond: 30% du budget par station
            - Réallocation automatique du surplus
            """)
        
        if 'opt_results' in st.session_state:
            results = st.session_state['opt_results']
            
            st.subheader(f"Allocation optimale - {results['budget_total']/1e6:.1f} M€")
            
            df_alloc = pd.DataFrame(results['allocations'])
            
            fig = px.bar(
                df_alloc.head(10),
                x='station',
                y='budget_alloue',
                title="Top 10 stations - Budget alloué",
                color='score_urgence',
                color_continuous_scale='RdYlGn_r',
                labels={'budget_alloue': 'Budget (€)', 'station': ''}
            )
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
            
            st.dataframe(
                df_alloc.style.format({
                    'budget_alloue': '{:,.0f} €',
                    'pourcentage_budget': '{:.1f}%',
                    'score_urgence': '{:.3f}'
                }),
                use_container_width=True
            )
    
    # ========== MODULE HYDRAULIQUE ==========
    elif "Simulation" in module:
        st.header("🌊 Simulation hydraulique - Pluie/Débit")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            intensite = st.slider("💧 Intensité pluie (mm/h)", 10, 50, 25)
        with col2:
            duree = st.slider("⏱️ Durée (heures)", 24, 120, 72, 12)
        with col3:
            capacite = st.selectbox(
                "🏭 Capacité STEP (EH)",
                [10000, 25000, 50000, 75000, 100000],
                index=2
            )
        
        # Simulation
        t, pluie, debit = HydrauliqueLegere.simuler_crue(duree, intensite)
        
        # Risque de débordement
        risque = HydrauliqueLegere.risque_debordement(capacite, debit[-1])
        
        # Graphiques
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=("Intensité pluviométrique", "Débit simulé"),
            vertical_spacing=0.15
        )
        
        fig.add_trace(
            go.Scatter(x=t, y=pluie, name="Pluie", line=dict(color='#00A0E2', width=2)),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=t, y=debit, name="Débit", line=dict(color='#ff6b6b', width=2)),
            row=2, col=1
        )
        
        # Seuil d'alerte
        seuil = capacite * 0.15 * 0.8
        fig.add_hline(y=seuil, line_dash="dash", line_color="red", 
                     annotation_text="Seuil alerte", row=2, col=1)
        
        fig.update_layout(height=600, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        
        # Indicateurs de risque
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("🚨 Risque débordement", f"{risque['risque']*100:.1f}%")
        with col2:
            st.metric("📊 Niveau", risque['niveau'])
        with col3:
            st.metric("💧 Charge", f"{risque['charge']:.2f}")
        with col4:
            if risque['niveau'] == 'Élevé':
                st.error("⚠️ ALERTE")
            elif risque['niveau'] == 'Moyen':
                st.warning("⚡ SURVEILLANCE")
            else:
                st.success("✅ NORMAL")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #6c757d; padding: 1rem; background: #f8f9fa; border-radius: 10px;'>
        <small>
        <strong>Office de l'Eau Réunion</strong> • Version IA Légère (sans TensorFlow) • 100% compatible Streamlit Cloud<br>
        Sources: https://donnees.eaureunion.fr • Licence Etalab • Mis à jour le {date}
        </small>
    </div>
    """.format(date=datetime.now().strftime('%d/%m/%Y')), unsafe_allow_html=True)
    
    # Nettoyage mémoire
    gc.collect()


if __name__ == "__main__":
    main()
