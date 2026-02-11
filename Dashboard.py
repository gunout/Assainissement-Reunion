# ------------------------------------------------------------------
# dashboard_assainissement_REUNION_OPEN_DATA.py
# Intégration officielle des données de l'Office de l'Eau Réunion
# Source : https://donnees.eaureunion.fr
# ------------------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import requests
from io import BytesIO, StringIO
import zipfile
import tempfile
import os
import base64

# Configuration de la page
st.set_page_config(
    page_title="Assainissement Réunion - Données Officielles",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================================
# 1️⃣ TÉLÉCHARGEMENT DIRECT DES DONNÉES OFFICIELLES
# ==========================================================
class DonneesOfficeEau:
    """Télécharge et prépare les données depuis donnees.eaureunion.fr"""
    
    # URLs officielles des fichiers
    URLS = {
        'stations_epuration': "https://donnees.eaureunion.fr/api/explore/v2.1/catalog/datasets/stations-de-traitement-des-eaux-usees/exports/csv",
        'systemes_alimentation': "https://donnees.eaureunion.fr/api/explore/v2.1/catalog/datasets/systemes-dalimentation-des-stations-de-traitement/exports/csv",
        'qualite_cours_eau': "https://donnees.eaureunion.fr/api/explore/v2.1/catalog/datasets/chimie-des-cours-deau/exports/csv",
        'debits': "https://donnees.eaureunion.fr/api/explore/v2.1/catalog/datasets/debit-moyen-journalier/exports/csv",
        'piezometrie': "https://donnees.eaureunion.fr/api/explore/v2.1/catalog/datasets/piezometrie-instantanee/exports/csv"
    }
    
    @staticmethod
    @st.cache_data(ttl=3600)  # Cache 1h pour respecter les serveurs
    def telecharger_stations_epuration():
        """
        Télécharge la liste officielle des STEP de La Réunion
        Source : Stations de traitement des eaux usées
        """
        try:
            url = DonneesOfficeEau.URLS['stations_epuration']
            response = requests.get(url, timeout=30)
            response.encoding = 'utf-8'
            
            if response.status_code == 200:
                df = pd.read_csv(StringIO(response.text), sep=';')
                
                # Renommage des colonnes pour plus de clarté
                df = df.rename(columns={
                    'libelle_de_la_station': 'nom_station',
                    'code_commune': 'commune_code',
                    'nom_commune': 'commune',
                    'filiere_de_traitement': 'filiere',
                    'capacite_nominale_en_eh': 'capacite_eh',
                    'annee_de_mise_en_service': 'mise_service',
                    'code_masse_eau_rejet': 'code_masse_eau',
                    'statut': 'statut'
                })
                
                return df
            else:
                st.error(f"Erreur téléchargement STEP: {response.status_code}")
                return DonneesOfficeEau._get_stations_exemple()
                
        except Exception as e:
            st.warning(f"Utilisation des données de démonstration: {str(e)}")
            return DonneesOfficeEau._get_stations_exemple()
    
    @staticmethod
    def _get_stations_exemple():
        """Données de secours au cas où le téléchargement échoue"""
        return pd.DataFrame({
            'nom_station': ['STEP Saint-Denis', 'STEP Saint-Paul', 'STEP Saint-Pierre'],
            'commune': ['Saint-Denis', 'Saint-Paul', 'Saint-Pierre'],
            'filiere': ['Boues activées', 'Lagunage', 'Boues activées'],
            'capacite_eh': [85000, 62000, 48000],
            'mise_service': [1998, 2005, 2008],
            'statut': ['En service', 'En service', 'En service']
        })
    
    @staticmethod
    @st.cache_data(ttl=3600)
    def telecharger_qualite_cours_eau(limite_lignes=10000):
        """
        Données de chimie des cours d'eau (limitées pour performance)
        Source : Chimie des cours d'eau (>20 Mo)
        """
        try:
            url = DonneesOfficeEau.URLS['qualite_cours_eau']
            response = requests.get(f"{url}&limit={limite_lignes}", timeout=45)
            
            if response.status_code == 200:
                df = pd.read_csv(StringIO(response.text), sep=';')
                return df
            else:
                return pd.DataFrame()
                
        except:
            return pd.DataFrame()
    
    @staticmethod
    @st.cache_data(ttl=1800)
    def telecharger_debits_journaliers():
        """
        Débits moyens journaliers des rivières
        Source : Débit moyen journalier
        """
        try:
            url = DonneesOfficeEau.URLS['debits']
            response = requests.get(f"{url}&limit=5000", timeout=30)
            
            if response.status_code == 200:
                df = pd.read_csv(StringIO(response.text), sep=';')
                return df
            else:
                return pd.DataFrame()
                
        except:
            return pd.DataFrame()


# ==========================================================
# 2️⃣ CHARGEMENT DES DONNÉES COMMUNALES
# ==========================================================
class ReferentielCommunes:
    """Gestion du référentiel des 24 communes"""
    
    @staticmethod
    def get_liste_officielle():
        """Retourne la liste officielle des communes avec codes INSEE"""
        return {
            '97411': 'Saint-Denis',
            '97415': 'Saint-Paul', 
            '97416': 'Saint-Pierre',
            '97422': 'Le Tampon',
            '97409': 'Saint-André',
            '97414': 'Sainte-Marie',
            '97413': 'Saint-Louis',
            '97407': 'Le Port',
            '97410': 'Saint-Benoît',
            '97412': 'Saint-Joseph',
            '97420': 'Sainte-Suzanne',
            '97413': 'Saint-Leu',
            '97408': 'La Possession',
            '97402': 'Bras-Panon',
            '97401': 'Les Avirons',
            '97404': 'Cilaos',
            '97403': 'Entre-Deux',
            '97405': 'L\'Étang-Salé',
            '97406': 'Petite-Île',
            '97417': 'La Plaine-des-Palmistes',
            '97418': 'Saint-Philippe',
            '97419': 'Sainte-Rose',
            '97421': 'Salazie',
            '97423': 'Trois-Bassins'
        }
    
    @staticmethod
    def get_commune_par_code(code):
        communes = ReferentielCommunes.get_liste_officielle()
        return communes.get(str(code), 'Non trouvée')


# ==========================================================
# 3️⃣ MODULE D'ANALYSE SPÉCIFIQUE À L'ASSAINISSEMENT
# ==========================================================
class AnalyseAssainissement:
    """Métriques et indicateurs à partir des données Office de l'Eau"""
    
    @staticmethod
    def indicateurs_commune(df_stations, commune):
        """Calcule les KPIs pour une commune donnée"""
        
        stations_commune = df_stations[df_stations['commune'].str.contains(commune, na=False)]
        
        if stations_commune.empty:
            return None
        
        indicateurs = {
            'nb_stations': len(stations_commune),
            'capacite_totale_eh': stations_commune['capacite_eh'].sum(),
            'filiere_principale': stations_commune['filiere'].mode().iloc[0] if not stations_commune['filiere'].mode().empty else 'Non renseigné',
            'annee_moyenne_mise_service': int(stations_commune['mise_service'].mean()),
            'stations': stations_commune.to_dict('records')
        }
        
        return indicateurs
    
    @staticmethod
    def comparer_communes(df_stations):
        """Tableau de comparaison inter-communal"""
        comparaison = df_stations.groupby('commune').agg({
            'capacite_eh': ['sum', 'mean', 'count'],
            'mise_service': 'mean'
        }).round(0)
        
        comparaison.columns = ['capacite_totale', 'capacite_moyenne', 'nb_stations', 'annee_moyenne']
        return comparaison.sort_values('capacite_totale', ascending=False)


# ==========================================================
# 4️⃣ INTERFACE STREAMLIT
# ==========================================================
class DashboardOpenDataReunion:
    def __init__(self):
        self.init_session()
        self.charger_donnees_officielles()
    
    def init_session(self):
        """Initialise la session Streamlit"""
        if 'df_stations' not in st.session_state:
            st.session_state.df_stations = None
        if 'df_debits' not in st.session_state:
            st.session_state.df_debits = None
        if 'commune_active' not in st.session_state:
            st.session_state.commune_active = 'Saint-Denis'
        if 'derniere_maj' not in st.session_state:
            st.session_state.derniere_maj = None
    
    def charger_donnees_officielles(self):
        """Charge les données depuis l'Office de l'Eau"""
        with st.spinner("📡 Téléchargement des données officielles depuis donnees.eaureunion.fr..."):
            
            # Téléchargement des STEP
            st.session_state.df_stations = DonneesOfficeEau.telecharger_stations_epuration()
            
            # Téléchargement des débits (optionnel)
            st.session_state.df_debits = DonneesOfficeEau.telecharger_debits_journaliers()
            
            # Timestamp
            st.session_state.derniere_maj = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    
    def afficher_en_tete(self):
        """En-tête avec source officielle"""
        st.markdown("""
        <div style='text-align: center; padding: 1rem; background: linear-gradient(135deg, #0066B3 0%, #00A0E2 100%); border-radius: 15px; margin-bottom: 2rem;'>
            <h1 style='color: white; font-size: 2.2rem;'>💧 OFFICE DE L'EAU RÉUNION</h1>
            <p style='color: white; font-size: 1.1rem;'>Données Open Data - Stations de traitement des eaux usées</p>
            <p style='color: #FFD700; font-size: 0.9rem;'>Source: donnees.eaureunion.fr • Mise à jour: {}</p>
        </div>
        """.format(st.session_state.derniere_maj), unsafe_allow_html=True)
    
    def afficher_selecteur_commune(self):
        """Sélecteur de commune basé sur les données réelles"""
        if st.session_state.df_stations is not None:
            communes_disponibles = st.session_state.df_stations['commune'].dropna().unique()
            communes_triees = sorted([c for c in communes_disponibles if c != 'Non renseigné'])
        else:
            communes_triees = list(ReferentielCommunes.get_liste_officielle().values())
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            selected = st.selectbox(
                "🔍 Sélectionnez une commune",
                communes_triees,
                index=communes_triees.index(st.session_state.commune_active) 
                if st.session_state.commune_active in communes_triees else 0
            )
            st.session_state.commune_active = selected
        
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🔄 Rafraîchir", use_container_width=True):
                self.charger_donnees_officielles()
                st.rerun()
    
    def afficher_kpis_commune(self):
        """Indicateurs clés pour la commune sélectionnée"""
        
        if st.session_state.df_stations is None:
            st.warning("Données non disponibles")
            return
        
        indicateurs = AnalyseAssainissement.indicateurs_commune(
            st.session_state.df_stations, 
            st.session_state.commune_active
        )
        
        if not indicateurs:
            st.info(f"Aucune station d'épuration recensée pour {st.session_state.commune_active}")
            return
        
        # KPIS
        cols = st.columns(4)
        
        with cols[0]:
            st.metric(
                "🏭 Stations d'épuration",
                indicateurs['nb_stations'],
                delta=None
            )
        
        with cols[1]:
            st.metric(
                "👥 Capacité totale",
                f"{indicateurs['capacite_totale_eh']:,.0f} EH",
                delta=None
            )
        
        with cols[2]:
            st.metric(
                "⚙️ Filière principale",
                indicateurs['filiere_principale'],
                delta=None
            )
        
        with cols[3]:
            st.metric(
                "📅 Année moyenne",
                indicateurs['annee_moyenne_mise_service'],
                delta=None
            )
        
        # Détail des stations
        st.subheader(f"📋 Stations de la commune")
        
        for station in indicateurs['stations']:
            with st.expander(f"🏭 {station['nom_station']}"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Capacité :** {station['capacite_eh']:,.0f} EH")
                    st.write(f"**Filière :** {station['filiere']}")
                with col2:
                    st.write(f"**Mise en service :** {station['mise_service']}")
                    st.write(f"**Statut :** {station.get('statut', 'En service')}")
    
    def afficher_carte_virtuelle(self):
        """Visualisation des capacités par commune"""
        st.subheader("🗺️ Répartition régionale des capacités de traitement")
        
        if st.session_state.df_stations is not None:
            
            df_capa = st.session_state.df_stations.groupby('commune').agg({
                'capacite_eh': 'sum',
                'nom_station': 'count'
            }).reset_index()
            
            df_capa = df_capa.sort_values('capacite_eh', ascending=True)
            
            fig = px.bar(
                df_capa.tail(10),
                x='capacite_eh',
                y='commune',
                orientation='h',
                title="Top 10 communes - Capacité totale STEP",
                labels={'capacite_eh': 'Capacité (EH)', 'commune': ''},
                color='capacite_eh',
                color_continuous_scale='Blues',
                text='nom_station'
            )
            
            fig.update_traces(texttemplate='%{text} station(s)', textposition='outside')
            fig.update_layout(height=500)
            
            st.plotly_chart(fig, use_container_width=True)
    
    def afficher_donnees_brutes(self):
        """Accès aux données sources"""
        with st.expander("📁 Accès aux données brutes (CSV officiel)"):
            
            if st.session_state.df_stations is not None:
                
                csv = st.session_state.df_stations.to_csv(index=False, sep=';').encode('utf-8')
                b64 = base64.b64encode(csv).decode()
                
                st.markdown(f"""
                <div style='background: #f8f9fa; padding: 1rem; border-radius: 10px;'>
                    <h5>Fichier: Stations de traitement des eaux usées</h5>
                    <p>Source: <a href='https://donnees.eaureunion.fr' target='_blank'>Office de l'Eau Réunion</a></p>
                    <p>Lignes: {len(st.session_state.df_stations)} | Colonnes: {len(st.session_state.df_stations.columns)}</p>
                    <a href='data:text/csv;base64,{b64}' download='stations_epuration_reunion.csv' style='background: #0066B3; color: white; padding: 0.5rem 1rem; border-radius: 5px; text-decoration: none;'>📥 Télécharger CSV complet</a>
                </div>
                """, unsafe_allow_html=True)
                
                st.dataframe(st.session_state.df_stations.head(20), use_container_width=True)
    
    def afficher_comparaison(self):
        """Tableau de comparaison inter-communes"""
        st.subheader("📊 Comparaison inter-communale")
        
        if st.session_state.df_stations is not None:
            comparaison = AnalyseAssainissement.comparer_communes(st.session_state.df_stations)
            st.dataframe(comparaison.style.format({
                'capacite_totale': '{:,.0f}',
                'capacite_moyenne': '{:,.0f}',
                'annee_moyenne': '{:.0f}'
            }), use_container_width=True)
    
    def afficher_footer(self):
        """Mentions légales et sources"""
        st.markdown("---")
        st.markdown(f"""
        <div style='text-align: center; color: #6c757d; font-size: 0.8rem; padding: 1rem;'>
            <strong>Données officielles - Office de l'Eau Réunion</strong><br>
            Licence Ouverte / Open Licence - Version 2.0 - Etalab<br>
            Ces données sont téléchargées en temps réel depuis <a href='https://donnees.eaureunion.fr/page-opendata/'>donnees.eaureunion.fr</a><br>
            Traitement effectué le {st.session_state.derniere_maj}
        </div>
        """, unsafe_allow_html=True)
    
    def run(self):
        """Exécution principale"""
        self.afficher_en_tete()
        self.afficher_selecteur_commune()
        self.afficher_kpis_commune()
        self.afficher_carte_virtuelle()
        self.afficher_comparaison()
        self.afficher_donnees_brutes()
        self.afficher_footer()


# ==========================================================
# 5️⃣ LANCEMENT
# ==========================================================
if __name__ == "__main__":
    dashboard = DashboardOpenDataReunion()
    dashboard.run()
