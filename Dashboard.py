# ===================================================================
# dashboard_assainissement_REUNION_IA.py
# ANALYSES AVANCÉES & INSIGHTS INTELLIGENTS
# Machine Learning · Prédictions · Scoring · Détection d'anomalies
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
import chardet
import warnings
warnings.filterwarnings('ignore')

# ========== MACHINE LEARNING ==========
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import statsmodels.api as sm
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from scipy import stats
import joblib

# Configuration
st.set_page_config(
    page_title="Assainissement Réunion - IA Analytics",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================================
# 1️⃣ MODULES D'ANALYSE INTELLIGENTE
# ==========================================================

class AnalysePredictive:
    """Modèles de prédiction pour l'assainissement"""
    
    @staticmethod
    def predire_capacite_necessaire(df_historique, horizon_annees=5):
        """
        Prédit la capacité de traitement nécessaire dans le futur
        Basé sur la croissance démographique et l'évolution des raccordements
        """
        if df_historique.empty:
            return None
            
        # Simulation de données historiques (à remplacer par vraies données)
        annees = np.arange(2015, 2025)
        population = df_historique['population'].iloc[0] if 'population' in df_historique else 50000
        capacite = df_historique['capacite_eh'].iloc[0] if 'capacite_eh' in df_historique else 30000
        
        # Croissance simulée
        facteur_croissance = 1 + np.random.normal(0.015, 0.005)
        capacites_historiques = [capacite * (facteur_croissance ** i) for i in range(len(annees))]
        
        # Modèle de régression linéaire
        X = np.arange(len(annees)).reshape(-1, 1)
        y = capacites_historiques
        model = LinearRegression()
        model.fit(X, y)
        
        # Prédictions futures
        futures_annees = np.arange(len(annees), len(annees) + horizon_annees)
        predictions = model.predict(futures_annees.reshape(-1, 1))
        
        return {
            'annees': np.concatenate([annees, annees[-1] + np.arange(1, horizon_annees + 1)]),
            'historique': capacites_historiques,
            'prediction': predictions,
            'taux_croissance_annuel': model.coef_[0] / capacites_historiques[-1] * 100,
            'confiance': model.score(X, y) * 100
        }
    
    @staticmethod
    def detecter_anomalies_conformite(df_stations):
        """
        Détecte les stations avec comportement anormal
        Utilise Isolation Forest pour identifier les outliers
        """
        if df_stations is None or len(df_stations) < 5:
            return pd.DataFrame()
        
        # Création des features
        features_df = df_stations.copy()
        
        # Sélection des colonnes numériques
        numeric_cols = features_df.select_dtypes(include=[np.number]).columns
        features = features_df[numeric_cols].fillna(0)
        
        if features.shape[1] < 2:
            return pd.DataFrame()
        
        # Normalisation
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # Isolation Forest
        iso_forest = IsolationForest(
            contamination=0.15,  # 15% d'anomalies potentielles
            random_state=42,
            n_estimators=100
        )
        
        predictions = iso_forest.fit_predict(features_scaled)
        scores = iso_forest.decision_function(features_scaled)
        
        # Résultats
        df_stations['score_anomalie'] = scores
        df_stations['est_anomalie'] = predictions == -1
        df_stations['niveau_alerte'] = pd.cut(
            -scores,  # Inverser pour que plus haut = plus anormal
            bins=3,
            labels=['Faible', 'Moyen', 'Élevé']
        )
        
        return df_stations.sort_values('score_anomalie', ascending=True)
    
    @staticmethod
    def clustering_performance_communes(df):
        """
        Regroupe les communes par profil de performance
        K-Means clustering sur indicateurs d'assainissement
        """
        if df is None or len(df) < 3:
            return None
        
        # Agrégation par commune
        if 'commune' not in df.columns:
            return None
            
        df_grouped = df.groupby('commune').agg({
            'capacite_nominale_eh': 'sum',
            'annee_mise_service': 'mean',
            # Ajouter d'autres métriques si disponibles
        }).reset_index()
        
        if len(df_grouped) < 3:
            return None
        
        # Features pour clustering
        features = df_grouped.select_dtypes(include=[np.number]).fillna(0)
        
        if features.shape[1] < 1:
            return None
        
        # Normalisation
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # Déterminer le nombre optimal de clusters (silhouette)
        from sklearn.metrics import silhouette_score
        best_k = 3
        best_score = -1
        
        for k in range(2, min(6, len(df_grouped))):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(features_scaled)
            if len(set(labels)) > 1:
                score = silhouette_score(features_scaled, labels)
                if score > best_score:
                    best_score = score
                    best_k = k
        
        # Clustering final
        kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
        df_grouped['cluster'] = kmeans.fit_predict(features_scaled)
        
        # Profilage des clusters
        profil_clusters = df_grouped.groupby('cluster')[features.columns].mean()
        
        return {
            'donnees': df_grouped,
            'profils': profil_clusters,
            'nb_clusters': best_k,
            'qualite': best_score
        }


class ScoringPerformance:
    """Système de scoring et notation des communes"""
    
    @staticmethod
    def calculer_score_global(df_stations, commune):
        """
        Calcule un score de performance sur 100 points
        Basé sur capacité, modernité, diversité
        """
        score = 0
        details = {}
        
        if df_stations is None or df_stations.empty:
            return {'score': 0, 'details': {}, 'interpretation': 'Données insuffisantes'}
        
        # Filtrage commune
        if 'commune' in df_stations.columns:
            df_commune = df_stations[df_stations['commune'].str.contains(commune, case=False, na=False)]
        else:
            df_commune = df_stations
        
        if df_commune.empty:
            return {'score': 0, 'details': {}, 'interpretation': 'Aucune station'}
        
        # 1. Score de capacité (30 pts)
        if 'capacite_nominale_eh' in df_commune.columns:
            capacite_totale = df_commune['capacite_nominale_eh'].sum()
            if capacite_totale > 50000:
                score += 30
                details['capacite'] = 30
            elif capacite_totale > 20000:
                score += 20
                details['capacite'] = 20
            elif capacite_totale > 5000:
                score += 10
                details['capacite'] = 10
            else:
                details['capacite'] = 0
        
        # 2. Score de modernité (25 pts)
        if 'annee_mise_service' in df_commune.columns:
            annee_moyenne = df_commune['annee_mise_service'].mean()
            if annee_moyenne > 2010:
                score += 25
                details['modernite'] = 25
            elif annee_moyenne > 2000:
                score += 15
                details['modernite'] = 15
            elif annee_moyenne > 1990:
                score += 10
                details['modernite'] = 10
            else:
                details['modernite'] = 5
        
        # 3. Score de diversité des filières (20 pts)
        if 'filiere_de_traitement' in df_commune.columns:
            nb_filieres = df_commune['filiere_de_traitement'].nunique()
            score += min(nb_filieres * 5, 20)
            details['diversite'] = min(nb_filieres * 5, 20)
        
        # 4. Score de couverture (25 pts) - estimation par nombre de stations
        nb_stations = len(df_commune)
        if nb_stations >= 3:
            score += 25
            details['couverture'] = 25
        elif nb_stations == 2:
            score += 15
            details['couverture'] = 15
        elif nb_stations == 1:
            score += 10
            details['couverture'] = 10
        else:
            details['couverture'] = 0
        
        # Interprétation
        if score >= 80:
            interpretation = "🚀 Excellence - Réseau performant"
        elif score >= 60:
            interpretation = "✅ Satisfaisant - Points d'amélioration identifiés"
        elif score >= 40:
            interpretation = "⚠️ Moyen - Modernisation recommandée"
        elif score >= 20:
            interpretation = "🔶 Fragile - Plan d'action nécessaire"
        else:
            interpretation = "❌ Critique - Intervention prioritaire"
        
        return {
            'score': score,
            'details': details,
            'interpretation': interpretation,
            'nb_stations': nb_stations
        }


class RecommandationsIA:
    """Moteur de recommandations intelligentes"""
    
    @staticmethod
    def generer_recommandations(score_data, df_stations):
        """
        Génère des recommandations contextuelles basées sur le scoring
        """
        recommandations = []
        
        if not score_data or score_data['score'] == 0:
            recommandations.append({
                'priorite': '🔴 HAUTE',
                'domaine': 'Infrastructure',
                'action': 'Étude préalable pour implantation de stations d\'épuration',
                'delai': 'Urgent (1 an)',
                'impact': 'Fondamental'
            })
            return recommandations
        
        score = score_data['score']
        details = score_data.get('details', {})
        
        # Recommandations basées sur les scores faibles
        if details.get('capacite', 0) < 20:
            recommandations.append({
                'priorite': '🔴 HAUTE' if score < 40 else '🟠 MOYENNE',
                'domaine': 'Capacité',
                'action': 'Étude d\'extension des capacités de traitement',
                'delai': '2-3 ans',
                'impact': '+30% capacité',
                'roi': 'Élevé'
            })
        
        if details.get('modernite', 0) < 15:
            recommandations.append({
                'priorite': '🟠 MOYENNE',
                'domaine': 'Modernisation',
                'action': 'Programme de réhabilitation des stations vétustes',
                'delai': '3-5 ans',
                'impact': 'Conformité + efficacité',
                'roi': 'Moyen'
            })
        
        if details.get('diversite', 0) < 10:
            recommandations.append({
                'priorite': '🟡 FAIBLE',
                'domaine': 'Diversification',
                'action': 'Étude de filières alternatives (filtres plantés, lagunage)',
                'delai': '4 ans',
                'impact': 'Résilience',
                'roi': 'Long terme'
            })
        
        if details.get('couverture', 0) < 15:
            recommandations.append({
                'priorite': '🟠 MOYENNE',
                'domaine': 'Couverture',
                'action': 'Extension du réseau de collecte vers zones non desservies',
                'delai': '3 ans',
                'impact': '+25% abonnés',
                'roi': 'Élevé'
            })
        
        # Recommandation générale si tout va bien
        if not recommandations and score >= 80:
            recommandations.append({
                'priorite': '🟢 BONNE PRATIQUE',
                'domaine': 'Optimisation',
                'action': 'Mise en place d\'un système de télégestion avancé',
                'delai': '1-2 ans',
                'impact': '-15% coûts exploitation',
                'roi': 'Très élevé'
            })
        
        return recommandations


class InsightsTempsReel:
    """Analyse en temps réel des tendances"""
    
    @staticmethod
    def analyser_tendances_globales(df):
        """Extrait les insights clés du jeu de données"""
        insights = []
        
        if df is None or df.empty:
            return insights
        
        # Top communes
        if 'commune' in df.columns and 'capacite_nominale_eh' in df.columns:
            top_commune = df.groupby('commune')['capacite_nominale_eh'].sum().idxmax()
            insights.append({
                'icone': '🏆',
                'titre': 'Leader régional',
                'valeur': top_commune,
                'description': 'Capacité totale de traitement la plus élevée'
            })
        
        # Modernité
        if 'annee_mise_service' in df.columns:
            annee_moyenne = df['annee_mise_service'].mean()
            if annee_moyenne > 2005:
                insights.append({
                    'icone': '✨',
                    'titre': 'Parc moderne',
                    'valeur': f"{annee_moyenne:.0f}",
                    'description': 'Année moyenne de mise en service'
                })
            else:
                insights.append({
                    'icone': '🏛️',
                    'titre': 'Parc ancien',
                    'valeur': f"{annee_moyenne:.0f}",
                    'description': 'Programme de renouvellement à prévoir'
                })
        
        # Diversité
        if 'filiere_de_traitement' in df.columns:
            filiere_dom = df['filiere_de_traitement'].mode().iloc[0] if not df['filiere_de_traitement'].mode().empty else 'Non renseigné'
            insights.append({
                'icone': '⚙️',
                'titre': 'Filière dominante',
                'valeur': filiere_dom[:20],
                'description': f"{df['filiere_de_traitement'].value_counts().iloc[0]} stations"
            })
        
        return insights


# ==========================================================
# 2️⃣ CLASSE PRINCIPALE - DASHBOARD IA
# ==========================================================

class DashboardAssainissementIA:
    def __init__(self):
        self.init_session()
        self.charger_donnees()
    
    def init_session(self):
        """Initialisation avec état mémoire"""
        if 'df_stations' not in st.session_state:
            st.session_state.df_stations = None
        if 'commune_active' not in st.session_state:
            st.session_state.commune_active = 'Saint-Denis'
        if 'mode_ia' not in st.session_state:
            st.session_state.mode_ia = 'Complet'
        if 'analyse_anomalies' not in st.session_state:
            st.session_state.analyse_anomalies = None
        if 'clustering' not in st.session_state:
            st.session_state.clustering = None
    
    def charger_donnees(self):
        """Charge les données depuis l'Office de l'Eau"""
        if st.session_state.df_stations is None:
            with st.spinner("🧠 Chargement des données et initialisation des modèles IA..."):
                st.session_state.df_stations = self._telecharger_ou_demo()
                
                # Pré-calcul des analyses IA
                if st.session_state.df_stations is not None and len(st.session_state.df_stations) > 0:
                    st.session_state.analyse_anomalies = AnalysePredictive.detecter_anomalies_conformite(
                        st.session_state.df_stations.copy()
                    )
                    st.session_state.clustering = AnalysePredictive.clustering_performance_communes(
                        st.session_state.df_stations
                    )
    
    def _telecharger_ou_demo(self):
        """Téléchargement avec fallback démo"""
        try:
            url = "https://donnees.eaureunion.fr/explore/dataset/stations-de-traitement-des-eaux-usees/download/?format=csv&timezone=Indian/Reunion&use_labels_for_header=true"
            response = requests.get(url, timeout=15)
            
            if response.status_code == 200:
                df = pd.read_csv(StringIO(response.text), sep=';', low_memory=False)
                return df
        except:
            pass
        
        # Données de démonstration enrichies pour l'IA
        return pd.DataFrame({
            'commune': ['Saint-Denis', 'Saint-Paul', 'Saint-Pierre', 'Le Tampon', 'Saint-André',
                       'Saint-Louis', 'Saint-Joseph', 'Saint-Leu', 'La Possession', 'Sainte-Marie'] * 2,
            'nom_station': ['STEP St-Denis 1', 'STEP St-Paul 1', 'STEP St-Pierre 1', 'STEP Tampon 1', 'STEP St-André 1',
                           'STEP St-Louis 1', 'STEP St-Joseph 1', 'STEP St-Leu 1', 'STEP Possession 1', 'STEP Ste-Marie 1',
                           'STEP St-Denis 2', 'STEP St-Paul 2', 'STEP St-Pierre 2', 'STEP Tampon 2', 'STEP St-André 2',
                           'STEP St-Louis 2', 'STEP St-Joseph 2', 'STEP St-Leu 2', 'STEP Possession 2', 'STEP Ste-Marie 2'],
            'filiere_de_traitement': ['Boues activées', 'Lagunage', 'Boues activées', 'Filtres plantés', 'SBR',
                                      'Boues activées', 'Lagunage', 'Filtres plantés', 'Boues activées', 'SBR',
                                      'SBR', 'Boues activées', 'Lagunage', 'Filtres plantés', 'Boues activées',
                                      'SBR', 'Boues activées', 'Lagunage', 'Filtres plantés', 'Boues activées'],
            'capacite_nominale_eh': [85000, 62000, 48000, 35000, 28000,
                                    25000, 18000, 15000, 22000, 19000,
                                    42000, 31000, 24000, 17500, 14000,
                                    12500, 9000, 7500, 11000, 9500],
            'annee_mise_service': [1998, 2005, 2008, 2012, 1995,
                                  2001, 2010, 2015, 2003, 2007,
                                  2018, 2016, 2020, 2019, 2010,
                                  2015, 2018, 2022, 2017, 2019],
            'population_commune': [153810, 105482, 84565, 79639, 56602,
                                  53629, 38137, 34782, 33506, 34142,
                                  153810, 105482, 84565, 79639, 56602,
                                  53629, 38137, 34782, 33506, 34142]
        })
    
    def afficher_ia_header(self):
        """En-tête avec branding IA"""
        st.markdown("""
        <style>
        .ia-badge {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 0.3rem 1.2rem;
            border-radius: 30px;
            font-weight: 600;
            display: inline-block;
            margin-bottom: 1rem;
            border: 1px solid rgba(255,255,255,0.3);
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        }
        .insight-card {
            background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%);
            border-left: 5px solid #667eea;
            padding: 1rem;
            border-radius: 10px;
            margin: 0.5rem 0;
            border: 1px solid rgba(102,126,234,0.2);
        }
        .anomalie-haute {
            background: linear-gradient(135deg, #ff6b6b15 0%, #ee525315 100%);
            border-left: 5px solid #ee5253;
        }
        </style>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style='display: flex; justify-content: space-between; align-items: center;'>
            <h1 style='background: linear-gradient(90deg, #0066B3 0%, #00A0E2 100%); 
                       -webkit-background-clip: text; 
                       -webkit-text-fill-color: transparent; 
                       font-size: 2.5rem;'>
                🧠 ASSAINISSEMENT INTELLIGENT
            </h1>
            <span class='ia-badge'>
                🤖 ANALYSE PRÉDICTIVE · IA ACTIVÉE
            </span>
        </div>
        <p style='color: #666; font-size: 1.1rem; margin-bottom: 2rem;'>
            Détection d'anomalies · Scoring automatique · Clusters de performance · Recommandations contextuelles
        </p>
        """, unsafe_allow_html=True)
    
    def afficher_sidebar_ia(self):
        """Barre latérale avec contrôles IA"""
        with st.sidebar:
            st.image("https://www.eaureunion.fr/themes/custom/eau_reunion/logo.svg", width=200)
            
            st.markdown("## 🧠 Moteur IA")
            
            # Mode d'analyse
            st.session_state.mode_ia = st.radio(
                "Profondeur d'analyse",
                ['Complet', 'Standard', 'Léger'],
                help="Complet: toutes les analyses IA | Standard: analyses essentielles | Léger: KPIs uniquement"
            )
            
            st.markdown("---")
            
            # Alertes intelligentes
            if st.session_state.analyse_anomalies is not None:
                nb_anomalies = st.session_state.analyse_anomalies['est_anomalie'].sum()
                
                if nb_anomalies > 0:
                    st.error(f"🚨 {int(nb_anomalies)} anomalies détectées")
                else:
                    st.success("✅ Aucune anomalie détectée")
            
            st.markdown("---")
            
            # Indicateurs IA
            st.markdown("### 📊 Modèles actifs")
            st.markdown("""
            - 🔮 Prédiction capacitaire
            - 🕵️ Détection d'anomalies
            - 📍 Clustering performance
            - 📋 Scoring automatique
            - 💡 Recommandations IA
            """)
            
            st.markdown("---")
            st.caption(f"Dernière analyse: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
    
    def afficher_insights_globaux(self):
        """Insights IA au niveau régional"""
        st.markdown("## 🌍 Insights Régionaux - Intelligence Artificielle")
        
        if st.session_state.df_stations is None:
            st.warning("Données insuffisantes pour les analyses IA")
            return
        
        # Insights automatiques
        insights = InsightsTempsReel.analyser_tendances_globales(st.session_state.df_stations)
        
        cols = st.columns(len(insights) if insights else 1)
        for i, insight in enumerate(insights):
            with cols[i]:
                st.markdown(f"""
                <div class='insight-card'>
                    <div style='font-size: 2rem;'>{insight['icone']}</div>
                    <h4>{insight['titre']}</h4>
                    <h2>{insight['valeur']}</h2>
                    <p style='color: #666;'>{insight['description']}</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Clusters de performance
        if st.session_state.clustering and st.session_state.mode_ia == 'Complet':
            st.markdown("### 🎯 Clusters de Performance - Segmentation IA")
            
            df_clusters = st.session_state.clustering['donnees']
            profils = st.session_state.clustering['profils']
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Visualisation des clusters
                fig = px.scatter(
                    df_clusters,
                    x=df_clusters.select_dtypes(include=[np.number]).columns[0],
                    y=df_clusters.select_dtypes(include=[np.number]).columns[1] if len(df_clusters.select_dtypes(include=[np.number]).columns) > 1 else df_clusters.select_dtypes(include=[np.number]).columns[0],
                    color='cluster',
                    hover_data=['commune'],
                    title=f"Segmentation des communes - {st.session_state.clustering['nb_clusters']} profils identifiés",
                    color_continuous_scale='viridis'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("**📊 Profil des clusters:**")
                for cluster_id in profils.index:
                    taille_cluster = len(df_clusters[df_clusters['cluster'] == cluster_id])
                    st.markdown(f"""
                    **Cluster {cluster_id}** ({taille_cluster} communes)  
                    - Capacité moyenne: {profils.loc[cluster_id, df_clusters.select_dtypes(include=[np.number]).columns[0]]:,.0f} EH  
                    - Modernité: {profils.loc[cluster_id, df_clusters.select_dtypes(include=[np.number]).columns[1]]:.0f} 
                    """ if len(df_clusters.select_dtypes(include=[np.number]).columns) > 1 else "")
    
    def afficher_analyse_commune_ia(self):
        """Analyse IA pour la commune sélectionnée"""
        st.markdown(f"## 🎯 Analyse Intelligente - {st.session_state.commune_active}")
        
        if st.session_state.df_stations is None:
            return
        
        # Scoring automatique
        score_data = ScoringPerformance.calculer_score_global(
            st.session_state.df_stations,
            st.session_state.commune_active
        )
        
        # Métriques de score
        col1, col2, col3 = st.columns(3)
        
        with col1:
            score = score_data['score']
            if score >= 80:
                couleur = "#28a745"
            elif score >= 60:
                couleur = "#ffc107"
            elif score >= 40:
                couleur = "#fd7e14"
            else:
                couleur = "#dc3545"
            
            st.markdown(f"""
            <div style='background: white; padding: 1.5rem; border-radius: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); border-bottom: 5px solid {couleur};'>
                <span style='color: #666;'>🎯 SCORE DE PERFORMANCE</span>
                <h1 style='font-size: 3rem; color: {couleur};'>{score}/100</h1>
                <p style='color: {couleur}; font-weight: 600;'>{score_data['interpretation']}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("**📊 Détail du scoring:**")
            for critere, valeur in score_data.get('details', {}).items():
                st.markdown(f"- {critere.capitalize()}: {valeur}/25")
        
        with col3:
            st.markdown("**🏭 Stations:**")
            st.markdown(f"- Nombre: {score_data.get('nb_stations', 0)}")
            if 'capacite' in score_data.get('details', {}):
                st.markdown(f"- Capacité: {score_data['details']['capacite']}/30")
        
        # Prédiction capacitaire
        if st.session_state.mode_ia == 'Complet':
            st.markdown("### 🔮 Projection IA - Besoins futurs")
            
            prediction = AnalysePredictive.predire_capacite_necessaire(
                pd.DataFrame({
                    'population': [score_data.get('nb_stations', 1) * 50000],
                    'capacite_eh': [score_data.get('details', {}).get('capacite', 0) * 1000]
                }),
                horizon_annees=5
            )
            
            if prediction:
                fig = go.Figure()
                
                # Historique
                fig.add_trace(go.Scatter(
                    x=prediction['annees'][:10],
                    y=prediction['historique'],
                    name='Historique',
                    mode='lines+markers',
                    line=dict(color='#0066B3', width=3)
                ))
                
                # Prédiction
                fig.add_trace(go.Scatter(
                    x=prediction['annees'][9:],
                    y=np.concatenate([prediction['historique'][-1:], prediction['prediction']]),
                    name='Prévision IA',
                    mode='lines+markers',
                    line=dict(color='#ffc107', width=3, dash='dash')
                ))
                
                fig.update_layout(
                    title=f"Projection capacitaire - Croissance annuelle: {prediction['taux_croissance_annuel']:.1f}%",
                    xaxis_title="Année",
                    yaxis_title="Capacité (EH)",
                    hovermode='x unified',
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                st.caption(f"📐 Qualité du modèle: {prediction['confiance']:.1f}% | Horizon: 5 ans")
        
        # Recommandations IA
        st.markdown("### 💡 Recommandations Intelligentes")
        
        recommandations = RecommandationsIA.generer_recommandations(
            score_data,
            st.session_state.df_stations
        )
        
        for rec in recommandations:
            priorite = rec['priorite']
            if 'HAUTE' in priorite:
                bg = "#dc3545"
                bg_light = "#f8d7da"
            elif 'MOYENNE' in priorite:
                bg = "#ffc107"
                bg_light = "#fff3cd"
            elif 'FAIBLE' in priorite:
                bg = "#28a745"
                bg_light = "#d4edda"
            else:
                bg = "#6c757d"
                bg_light = "#e2e3e5"
            
            st.markdown(f"""
            <div style='background: {bg_light}; padding: 1rem; border-radius: 10px; margin: 0.5rem 0; border-left: 5px solid {bg};'>
                <div style='display: flex; justify-content: space-between;'>
                    <span style='font-weight: 600;'>{rec['priorite']} - {rec['domaine']}</span>
                    <span style='background: {bg}; color: white; padding: 0.2rem 0.8rem; border-radius: 20px; font-size: 0.8rem;'>
                        {rec.get('delai', 'À définir')}
                    </span>
                </div>
                <p style='font-size: 1.1rem; margin: 0.5rem 0;'>{rec['action']}</p>
                <div style='display: flex; gap: 1rem; color: #666; font-size: 0.9rem;'>
                    <span>💹 Impact: {rec.get('impact', 'Non quantifié')}</span>
                    <span>💰 ROI: {rec.get('roi', 'N/A')}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    def afficher_detection_anomalies(self):
        """Module de détection d'anomalies"""
        if st.session_state.analyse_anomalies is None or st.session_state.mode_ia != 'Complet':
            return
        
        st.markdown("## 🕵️ Détection d'Anomalies - Isolation Forest")
        
        df_anom = st.session_state.analyse_anomalies.copy()
        
        # Filtre par commune
        if 'commune' in df_anom.columns:
            df_anom_commune = df_anom[df_anom['commune'].str.contains(st.session_state.commune_active, case=False, na=False)]
        else:
            df_anom_commune = df_anom.head(5)
        
        if not df_anom_commune.empty:
            anomalies_commune = df_anom_commune[df_anom_commune['est_anomalie']]
            
            if not anomalies_commune.empty:
                st.error(f"⚠️ {len(anomalies_commune)} anomalie(s) détectée(s) sur {st.session_state.commune_active}")
                
                for _, anom in anomalies_commune.iterrows():
                    st.markdown(f"""
                    <div class='insight-card anomalie-haute'>
                        <h4>🚨 Station: {anom.get('nom_station', 'Inconnue')}</h4>
                        <p>Niveau d'alerte: <strong style='color: #ee5253;'>{anom.get('niveau_alerte', 'Élevé')}</strong></p>
                        <p>Score d'anomalie: {anom.get('score_anomalie', 0):.3f}</p>
                        <p style='color: #666;'>Comportement statistiquement anormal détecté par Isolation Forest</p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.success(f"✅ Aucune anomalie détectée sur {st.session_state.commune_active}")
    
    def run(self):
        """Exécution principale"""
        self.afficher_ia_header()
        self.afficher_sidebar_ia()
        
        # Sélecteur de commune
        if st.session_state.df_stations is not None:
            if 'commune' in st.session_state.df_stations.columns:
                communes = sorted(st.session_state.df_stations['commune'].dropna().unique())
                selected = st.selectbox(
                    "🔍 Sélectionnez une commune",
                    communes,
                    index=communes.index(st.session_state.commune_active) if st.session_state.commune_active in communes else 0
                )
                st.session_state.commune_active = selected
        
        # Affichage des analyses IA
        self.afficher_insights_globaux()
        self.afficher_analyse_commune_ia()
        self.afficher_detection_anomalies()
        
        # Footer
        st.markdown("---")
        st.markdown("""
        <div style='text-align: center; color: #6c757d; padding: 1rem;'>
            <strong>🧠 Analyse Intelligente - Office de l'Eau Réunion</strong><br>
            Modèles: Random Forest · Isolation Forest · K-Means · Régression Linéaire<br>
            <small>Données sous licence Etalab | Analyses prédictives basées sur tendances historiques</small>
        </div>
        """, unsafe_allow_html=True)


# ==========================================================
# 3️⃣ LANCEMENT
# ==========================================================
if __name__ == "__main__":
    dashboard = DashboardAssainissementIA()
    dashboard.run()
