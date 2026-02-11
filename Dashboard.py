# dashboard_assainissement_reunion_complet.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time
import random
import warnings
from functools import lru_cache
warnings.filterwarnings('ignore')

# Configuration de la page
st.set_page_config(
    page_title="Dashboard Assainissement - Toutes les communes de La Réunion",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        background: linear-gradient(45deg, #0066B3, #00A0E2, #0D5332);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .live-badge {
        background: linear-gradient(45deg, #0066B3, #00A0E2);
        color: white;
        padding: 0.3rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #0066B3;
        margin: 0.5rem 0;
    }
    .section-header {
        color: #0066B3;
        border-bottom: 2px solid #00A0E2;
        padding-bottom: 0.5rem;
        margin-top: 2rem;
    }
    .conforme { background-color: #d4edda; border-left: 4px solid #28a745; color: #155724; }
    .non-conforme { background-color: #f8d7da; border-left: 4px solid #dc3545; color: #721c24; }
    .en-travaux { background-color: #fff3cd; border-left: 4px solid #ffc107; color: #856404; }
    .station-card {
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 5px solid #0066B3;
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
    }
    .reunion-water {
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
        color: white;
        background: linear-gradient(90deg, #0066B3 33%, #00A0E2 33%, #00A0E2 66%, #0D5332 66%);
    }
    .alerte-haute { background-color: #dc3545; color: white; padding: 0.5rem; border-radius: 5px; }
    .alerte-moyenne { background-color: #ffc107; color: black; padding: 0.5rem; border-radius: 5px; }
    .alerte-faible { background-color: #28a745; color: white; padding: 0.5rem; border-radius: 5px; }
</style>
""", unsafe_allow_html=True)

# Initialisation de l'état de session
if 'assainissement_data' not in st.session_state:
    st.session_state.assainissement_data = {}
if 'selected_commune' not in st.session_state:
    st.session_state.selected_commune = 'SAINT_DENIS'
if 'last_update' not in st.session_state:
    st.session_state.last_update = datetime.now()

@st.cache_data(ttl=3600)
def get_communes_reunion():
    """Définit toutes les 24 communes de La Réunion"""
    return {
        # Villes principales
        'SAINT_DENIS': {
            'nom_complet': 'Saint-Denis',
            'population': 153810,
            'superficie': 14279,
            'prefecture': True,
            'categorie_taille': 'ville_principale',
            'nb_abonnes_eau': 42500,
            'nb_abonnes_assainissement': 38900,
            'taux_conformite': 87.5,
            'budget_assainissement': 32.5,
            'cours_eau_principal': 'Rivière Saint-Denis'
        },
        'SAINT_PAUL': {
            'nom_complet': 'Saint-Paul',
            'population': 105482,
            'superficie': 24128,
            'prefecture': False,
            'categorie_taille': 'grande_commune',
            'nb_abonnes_eau': 29800,
            'nb_abonnes_assainissement': 26500,
            'taux_conformite': 82.3,
            'budget_assainissement': 24.8,
            'cours_eau_principal': 'Rivière des Galets'
        },
        'SAINT_PIERRE': {
            'nom_complet': 'Saint-Pierre',
            'population': 84565,
            'superficie': 9553,
            'prefecture': False,
            'categorie_taille': 'grande_commune',
            'nb_abonnes_eau': 23500,
            'nb_abonnes_assainissement': 20800,
            'taux_conformite': 85.7,
            'budget_assainissement': 21.3,
            'cours_eau_principal': 'Rivière d\'Abord'
        },
        'LE_TAMPON': {
            'nom_complet': 'Le Tampon',
            'population': 79639,
            'superficie': 16542,
            'prefecture': False,
            'categorie_taille': 'grande_commune',
            'nb_abonnes_eau': 19800,
            'nb_abonnes_assainissement': 15200,
            'taux_conformite': 79.8,
            'budget_assainissement': 18.9,
            'cours_eau_principal': 'Bras de la Plaine'
        },
        'SAINT_ANDRE': {
            'nom_complet': 'Saint-André',
            'population': 56602,
            'superficie': 5355,
            'prefecture': False,
            'categorie_taille': 'moyenne_commune',
            'nb_abonnes_eau': 14800,
            'nb_abonnes_assainissement': 11200,
            'taux_conformite': 76.5,
            'budget_assainissement': 14.2,
            'cours_eau_principal': 'Rivière du Mât'
        },
        'SAINTE_MARIE': {
            'nom_complet': 'Sainte-Marie',
            'population': 34142,
            'superficie': 8726,
            'prefecture': False,
            'categorie_taille': 'moyenne_commune',
            'nb_abonnes_eau': 9500,
            'nb_abonnes_assainissement': 8200,
            'taux_conformite': 88.2,
            'budget_assainissement': 11.5,
            'cours_eau_principal': 'Rivière des Pluies'
        },
        'SAINT_LOUIS': {
            'nom_complet': 'Saint-Louis',
            'population': 53629,
            'superficie': 9873,
            'prefecture': False,
            'categorie_taille': 'moyenne_commune',
            'nb_abonnes_eau': 13200,
            'nb_abonnes_assainissement': 9800,
            'taux_conformite': 74.9,
            'budget_assainissement': 12.8,
            'cours_eau_principal': 'Rivière Saint-Étienne'
        },
        'LE_PORT': {
            'nom_complet': 'Le Port',
            'population': 32937,
            'superficie': 1675,
            'prefecture': False,
            'categorie_taille': 'moyenne_commune',
            'nb_abonnes_eau': 9800,
            'nb_abonnes_assainissement': 9100,
            'taux_conformite': 83.6,
            'budget_assainissement': 10.9,
            'cours_eau_principal': 'Rivière des Galets'
        },
        'SAINT_BENOIT': {
            'nom_complet': 'Saint-Benoît',
            'population': 37822,
            'superficie': 22961,
            'prefecture': False,
            'categorie_taille': 'moyenne_commune',
            'nb_abonnes_eau': 10100,
            'nb_abonnes_assainissement': 7200,
            'taux_conformite': 71.3,
            'budget_assainissement': 13.5,
            'cours_eau_principal': 'Rivière des Marsouins'
        },
        'SAINT_JOSEPH': {
            'nom_complet': 'Saint-Joseph',
            'population': 38137,
            'superficie': 17754,
            'prefecture': False,
            'categorie_taille': 'moyenne_commune',
            'nb_abonnes_eau': 9700,
            'nb_abonnes_assainissement': 6100,
            'taux_conformite': 68.7,
            'budget_assainissement': 11.2,
            'cours_eau_principal': 'Rivière Langevin'
        },
        'SAINTE_SUZANNE': {
            'nom_complet': 'Sainte-Suzanne',
            'population': 24714,
            'superficie': 5776,
            'prefecture': False,
            'categorie_taille': 'petite_commune',
            'nb_abonnes_eau': 7100,
            'nb_abonnes_assainissement': 5300,
            'taux_conformite': 79.4,
            'budget_assainissement': 8.7,
            'cours_eau_principal': 'Rivière Sainte-Suzanne'
        },
        'SAINT_LEU': {
            'nom_complet': 'Saint-Leu',
            'population': 34782,
            'superficie': 11809,
            'prefecture': False,
            'categorie_taille': 'moyenne_commune',
            'nb_abonnes_eau': 9200,
            'nb_abonnes_assainissement': 7400,
            'taux_conformite': 81.2,
            'budget_assainissement': 10.5,
            'cours_eau_principal': 'Ravine Saint-Gilles'
        },
        'LA_POSSESSION': {
            'nom_complet': 'La Possession',
            'population': 33506,
            'superficie': 1181,
            'prefecture': False,
            'categorie_taille': 'moyenne_commune',
            'nb_abonnes_eau': 9300,
            'nb_abonnes_assainissement': 8200,
            'taux_conformite': 77.8,
            'budget_assainissement': 9.8,
            'cours_eau_principal': 'Ravine à Marquet'
        },
        'BRAS_PANON': {
            'nom_complet': 'Bras-Panon',
            'population': 13477,
            'superficie': 887,
            'prefecture': False,
            'categorie_taille': 'petite_commune',
            'nb_abonnes_eau': 4200,
            'nb_abonnes_assainissement': 2800,
            'taux_conformite': 73.1,
            'budget_assainissement': 6.2,
            'cours_eau_principal': 'Rivière du Mât'
        },
        'LES_AVIRONS': {
            'nom_complet': 'Les Avirons',
            'population': 11513,
            'superficie': 2695,
            'prefecture': False,
            'categorie_taille': 'petite_commune',
            'nb_abonnes_eau': 3800,
            'nb_abonnes_assainissement': 2600,
            'taux_conformite': 75.6,
            'budget_assainissement': 5.9,
            'cours_eau_principal': 'Ravine Sèche'
        },
        'CILAOS': {
            'nom_complet': 'Cilaos',
            'population': 5538,
            'superficie': 8404,
            'prefecture': False,
            'categorie_taille': 'micro_commune',
            'nb_abonnes_eau': 2100,
            'nb_abonnes_assainissement': 1300,
            'taux_conformite': 82.5,
            'budget_assainissement': 5.1,
            'cours_eau_principal': 'Bras Rouge'
        },
        'ENTRE_DEUX': {
            'nom_complet': 'Entre-Deux',
            'population': 7098,
            'superficie': 6666,
            'prefecture': False,
            'categorie_taille': 'micro_commune',
            'nb_abonnes_eau': 2100,
            'nb_abonnes_assainissement': 1100,
            'taux_conformite': 69.8,
            'budget_assainissement': 4.8,
            'cours_eau_principal': 'Bras de Cilaos'
        },
        'L_ETANG_SALE': {
            'nom_complet': 'L\'Étang-Salé',
            'population': 14038,
            'superficie': 3893,
            'prefecture': False,
            'categorie_taille': 'petite_commune',
            'nb_abonnes_eau': 4600,
            'nb_abonnes_assainissement': 3800,
            'taux_conformite': 84.7,
            'budget_assainissement': 7.3,
            'cours_eau_principal': 'Ravine des Cafres'
        },
        'PETITE_ILE': {
            'nom_complet': 'Petite-Île',
            'population': 12395,
            'superficie': 3379,
            'prefecture': False,
            'categorie_taille': 'petite_commune',
            'nb_abonnes_eau': 3800,
            'nb_abonnes_assainissement': 2900,
            'taux_conformite': 72.4,
            'budget_assainissement': 6.1,
            'cours_eau_principal': 'Ravine des Cabris'
        },
        'LA_PLAINE_DES_PALMISTES': {
            'nom_complet': 'La Plaine-des-Palmistes',
            'population': 6664,
            'superficie': 8385,
            'prefecture': False,
            'categorie_taille': 'micro_commune',
            'nb_abonnes_eau': 2200,
            'nb_abonnes_assainissement': 1400,
            'taux_conformite': 77.3,
            'budget_assainissement': 4.9,
            'cours_eau_principal': 'Rivière des Marsouins'
        },
        'SAINT_PHILIPPE': {
            'nom_complet': 'Saint-Philippe',
            'population': 5088,
            'superficie': 15393,
            'prefecture': False,
            'categorie_taille': 'micro_commune',
            'nb_abonnes_eau': 1800,
            'nb_abonnes_assainissement': 900,
            'taux_conformite': 65.2,
            'budget_assainissement': 4.2,
            'cours_eau_principal': 'Rivière Langevin'
        },
        'SAINTE_ROSE': {
            'nom_complet': 'Sainte-Rose',
            'population': 6339,
            'superficie': 17756,
            'prefecture': False,
            'categorie_taille': 'micro_commune',
            'nb_abonnes_eau': 2000,
            'nb_abonnes_assainissement': 1100,
            'taux_conformite': 62.8,
            'budget_assainissement': 4.5,
            'cours_eau_principal': 'Rivière de l\'Est'
        },
        'SALAZIE': {
            'nom_complet': 'Salazie',
            'population': 7157,
            'superficie': 10328,
            'prefecture': False,
            'categorie_taille': 'micro_commune',
            'nb_abonnes_eau': 2100,
            'nb_abonnes_assainissement': 1200,
            'taux_conformite': 68.9,
            'budget_assainissement': 4.7,
            'cours_eau_principal': 'Rivière du Mât'
        },
        'TROIS_BASSINS': {
            'nom_complet': 'Trois-Bassins',
            'population': 7059,
            'superficie': 4285,
            'prefecture': False,
            'categorie_taille': 'micro_commune',
            'nb_abonnes_eau': 2100,
            'nb_abonnes_assainissement': 1500,
            'taux_conformite': 74.6,
            'budget_assainissement': 4.3,
            'cours_eau_principal': 'Ravine Trois-Bassins'
        }
    }

@st.cache_data(ttl=3600)
def get_infrastructures_assainissement(commune_code):
    """Définit les infrastructures d'assainissement pour une commune donnée"""
    communes = get_communes_reunion()
    commune_info = communes[commune_code]
    
    # Facteur d'ajustement selon la taille
    taille_factors = {
        'ville_principale': 1.0,
        'grande_commune': 0.85,
        'moyenne_commune': 0.7,
        'petite_commune': 0.55,
        'micro_commune': 0.4
    }
    factor = taille_factors.get(commune_info['categorie_taille'], 0.7)

    # Stations d'épuration
    nb_stations = {
        'ville_principale': random.randint(2, 4),
        'grande_commune': random.randint(1, 3),
        'moyenne_commune': random.randint(1, 2),
        'petite_commune': random.randint(0, 2),
        'micro_commune': random.randint(0, 1)
    }.get(commune_info['categorie_taille'], 1)

    stations = []
    for i in range(nb_stations):
        capacite = random.randint(5000, 35000) * factor
        stations.append({
            'nom': f"STEP {commune_info['nom_complet']} {i+1}",
            'type': random.choice(['Boues activées', 'Lagunage', 'Filtres plantés', 'SBR']),
            'capacite_nominale': capacite,
            'charge_actuelle': capacite * random.uniform(0.65, 0.95),
            'conformite': random.choice(['Conforme', 'Non conforme', 'En travaux']),
            'mise_en_service': random.randint(1995, 2020),
            'derniere_inspection': (datetime.now() - timedelta(days=random.randint(30, 365))).strftime('%d/%m/%Y')
        })

    # Postes de relevage
    nb_postes = int(15 * factor) + random.randint(-2, 5)
    postes = []
    for i in range(max(1, nb_postes)):
        postes.append({
            'nom': f"PR {commune_info['nom_complet']} {i+1}",
            'debit_m3h': random.randint(50, 500),
            'etat': random.choice(['Bon', 'Moyen', 'Dégradé']),
            'telemesure': random.choice([True, False]) if random.random() > 0.3 else True,
            'dernier_depannage': (datetime.now() - timedelta(days=random.randint(1, 180))).strftime('%d/%m/%Y') if random.random() > 0.5 else 'Aucun'
        })

    # Réseaux
    reseaux = {
        'EU': {
            'type': 'Eaux Usées',
            'longueur_km': commune_info['nb_abonnes_assainissement'] * 0.015 * factor * random.uniform(0.9, 1.1),
            'taux_renouvellement': random.uniform(0.5, 1.8),
            'rendement': random.uniform(75, 95),
            'nb_fuites_an': int(random.randint(5, 50) * factor),
            'couleur': '#0066B3'
        },
        'EP': {
            'type': 'Eaux Pluviales',
            'longueur_km': commune_info['nb_abonnes_eau'] * 0.008 * factor * random.uniform(0.8, 1.2),
            'taux_renouvellement': random.uniform(0.3, 1.2),
            'rendement': random.uniform(65, 90),
            'nb_fuites_an': int(random.randint(3, 30) * factor),
            'couleur': '#00A0E2'
        }
    }

    # SPANC - Assainissement Non Collectif
    spanc = {
        'nb_installations_ANC': int(commune_info['population'] * 0.15 * factor),
        'taux_conformite_ANC': random.uniform(55, 85),
        'nb_controles_annuels': int(commune_info['population'] * 0.05 * factor),
        'duree_moyenne_visite': random.randint(45, 90)
    }

    return {
        'stations': pd.DataFrame(stations),
        'postes_relevage': pd.DataFrame(postes),
        'reseaux': reseaux,
        'spanc': spanc
    }

@st.cache_data(ttl=1800)
def generate_historical_assainissement_data(commune_code, infra):
    """Génère les données historiques de la qualité des eaux et performances"""
    communes = get_communes_reunion()
    dates = pd.date_range('2020-01-01', datetime.now(), freq='M')
    data = []

    for date in dates:
        # Saisonnalité cyclonique : lessivage et surcharges
        if date.month in [1, 2, 3]:
            turbidity_factor = random.uniform(1.3, 1.8)
            conso_factor = random.uniform(0.9, 1.0)
        elif date.month in [7, 8, 9]:  # Hiver austral
            turbidity_factor = random.uniform(0.7, 0.9)
            conso_factor = random.uniform(1.0, 1.1)
        else:
            turbidity_factor = random.uniform(1.0, 1.2)
            conso_factor = random.uniform(0.95, 1.05)

        for station in infra['stations'].to_dict('records'):
            data.append({
                'date': date,
                'commune': commune_code,
                'infrastructure': station['nom'],
                'conformite': 1 if random.random() > 0.15 else 0,
                'db05_mgl': random.uniform(15, 35) * turbidity_factor,
                'dco_mgl': random.uniform(60, 150) * turbidity_factor,
                'mes_mgl': random.uniform(20, 60) * turbidity_factor,
                'ntk_mgl': random.uniform(15, 40) * turbidity_factor,
                'pt_mgl': random.uniform(2, 8) * turbidity_factor,
                'charge': random.uniform(65, 95) * conso_factor,
                'consommation_eau': communes[commune_code]['nb_abonnes_eau'] * 120 * conso_factor * random.uniform(0.95, 1.05)
            })

    return pd.DataFrame(data)

@st.cache_data(ttl=600)
def generate_travaux_assainissement(commune_code):
    """Génère les travaux en cours ou planifiés sur le réseau AEP/assainissement"""
    communes = get_communes_reunion()
    commune_info = communes[commune_code]

    travaux_types = [
        'Renouvellement collecteur EU',
        'Réhabilitation STEP',
        'Mise en séparatif',
        'Extension réseau',
        'Réfection regard',
        'Nettoyage bassin',
        'Télégestion poste',
        'Mise en conformité SPANC',
        'Branchements neufs'
    ]

    taille_travaux = {
        'ville_principale': (8, 18),
        'grande_commune': (6, 14),
        'moyenne_commune': (4, 10),
        'petite_commune': (2, 7),
        'micro_commune': (1, 4)
    }

    min_travaux, max_travaux = taille_travaux.get(commune_info['categorie_taille'], (2, 6))

    travaux_data = []
    for i in range(random.randint(min_travaux, max_travaux)):
        urgence = random.uniform(1, 10)
        if urgence > 8:
            statut = 'Urgent'
            couleur = '#dc3545'
        elif urgence > 5:
            statut = 'Planifié'
            couleur = '#ffc107'
        else:
            statut = 'En cours'
            couleur = '#28a745'

        travaux_data.append({
            'id_travail': f"AS_{commune_code}_{i+1:03d}",
            'type_travail': random.choice(travaux_types),
            'localisation': f"{random.choice(['Rue', 'Chemin', 'Avenue'])} {random.randint(1, 150)}",
            'urgence': urgence,
            'statut': statut,
            'couleur': couleur,
            'date_debut': (datetime.now() - timedelta(days=random.randint(0, 45))).strftime('%d/%m/%Y'),
            'duree_estimee': random.randint(10, 120),
            'budget': random.uniform(5, 250) * (1 if commune_info['categorie_taille'] == 'ville_principale' else 0.6),
            'impact_reseau': random.choice(['Faible', 'Moyen', 'Élevé']),
            'nb_abonnes_impactes': int(random.uniform(50, 2000) * (urgence/10))
        })

    return pd.DataFrame(travaux_data)

@st.cache_data(ttl=3600)
def generate_comparison_assainissement(communes):
    """Génère les données de comparaison entre communes sur l'assainissement"""
    comparison_data = []

    for commune_code, commune_info in communes.items():
        infra = get_infrastructures_assainissement(commune_code)
        
        # Calcul du taux de conformité global
        stations_conformes = infra['stations'][infra['stations']['conformite'] == 'Conforme'].shape[0]
        taux_conformite_stations = (stations_conformes / len(infra['stations'])) * 100 if len(infra['stations']) > 0 else 0
        
        # Rendement réseau EU
        rendement_eu = infra['reseaux']['EU']['rendement']
        
        comparison_data.append({
            'commune': commune_code,
            'nom_complet': commune_info['nom_complet'],
            'population': commune_info['population'],
            'categorie_taille': commune_info['categorie_taille'],
            'nb_abonnes_assainissement': commune_info['nb_abonnes_assainissement'],
            'taux_conformite_global': commune_info['taux_conformite'],
            'taux_conformite_stations': taux_conformite_stations,
            'nb_stations': len(infra['stations']),
            'nb_postes_relevage': len(infra['postes_relevage']),
            'rendement_reseau_eu': rendement_eu,
            'nb_fuites_an_eu': infra['reseaux']['EU']['nb_fuites_an'],
            'budget_assainissement': commune_info['budget_assainissement'],
            'taux_conformite_anc': infra['spanc']['taux_conformite_ANC'],
            'ratio_abonnes_reseau': (commune_info['nb_abonnes_assainissement'] / commune_info['population']) * 100
        })
    
    return pd.DataFrame(comparison_data)

class AssainissementReunionDashboard:
    def __init__(self):
        self.communes = get_communes_reunion()

    def get_commune_data(self, commune_code):
        """Récupère les données d'assainissement d'une commune avec cache"""
        if commune_code not in st.session_state.assainissement_data:
            with st.spinner(f"Chargement des données assainissement pour {self.communes[commune_code]['nom_complet']}..."):
                infra = get_infrastructures_assainissement(commune_code)
                historical_data = generate_historical_assainissement_data(commune_code, infra)
                travaux_data = generate_travaux_assainissement(commune_code)
                
                st.session_state.assainissement_data[commune_code] = {
                    'infra': infra,
                    'historical_data': historical_data,
                    'travaux_data': travaux_data,
                    'last_update': datetime.now()
                }
        
        return st.session_state.assainissement_data[commune_code]

    def update_live_data(self, commune_code):
        """Met à jour certaines données en temps réel (taux de charge, conformité)"""
        if commune_code in st.session_state.assainissement_data:
            data = st.session_state.assainissement_data[commune_code]
            
            # Mise à jour aléatoire des charges des stations
            for idx in data['infra']['stations'].index:
                if random.random() < 0.3:
                    current_charge = data['infra']['stations'].loc[idx, 'charge_actuelle']
                    variation = random.uniform(-0.03, 0.03)
                    data['infra']['stations'].loc[idx, 'charge_actuelle'] = max(0.4, min(1.0, current_charge * (1 + variation)))
            
            st.session_state.assainissement_data[commune_code]['last_update'] = datetime.now()

    def display_commune_selector(self):
        """Affiche le sélecteur de commune"""
        st.markdown('<div class="route-selector">', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            commune_options = {v['nom_complet']: k for k, v in self.communes.items()}
            current_name = self.communes[st.session_state.selected_commune]['nom_complet']
            selected_commune_name = st.selectbox(
                "💧 SÉLECTIONNEZ UNE COMMUNE:",
                options=list(commune_options.keys()),
                index=list(commune_options.keys()).index(current_name),
                key="commune_selector_assainissement"
            )
            new_commune = commune_options[selected_commune_name]
            if new_commune != st.session_state.selected_commune:
                st.session_state.selected_commune = new_commune
                self.get_commune_data(new_commune)
                st.success(f"✅ Chargement de {selected_commune_name} effectué!")
        
        with col2:
            commune_info = self.communes[st.session_state.selected_commune]
            st.metric("Abonnés EU", f"{commune_info['nb_abonnes_assainissement']:,}")
        with col3:
            st.metric("Taux conformité", f"{commune_info['taux_conformite']}%")
        
        st.markdown('</div>', unsafe_allow_html=True)

    def display_header(self):
        """Affiche l'en-tête du dashboard"""
        commune_info = self.communes[st.session_state.selected_commune]
        
        st.markdown(f'<h1 class="main-header">💧 Dashboard Assainissement - {commune_info["nom_complet"]}</h1>', 
                   unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown('<div class="live-badge">🔵 SURVEILLANCE ASSAINISSEMENT EN TEMPS RÉEL</div>', 
                       unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="reunion-water">
            <strong>ÎLE DE LA RÉUNION - Gestion de l'Eau et de l'Assainissement</strong><br>
            <small>{commune_info['nom_complet']} | Population: {commune_info['population']:,} | Cours d\'eau: {commune_info['cours_eau_principal']}</small>
        </div>
        """, unsafe_allow_html=True)
        
        current_time = datetime.now().strftime('%H:%M:%S')
        st.sidebar.markdown(f"**🕐 Dernière mise à jour: {current_time}**")

    def display_key_metrics(self):
        """Affiche les métriques clés de l'assainissement"""
        data = self.get_commune_data(st.session_state.selected_commune)
        commune_info = self.communes[st.session_state.selected_commune]
        infra = data['infra']
        
        st.markdown('<h3 class="section-header">📊 INDICATEURS DE PERFORMANCE ASSAINISSEMENT</h3>', 
                   unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            conformite_couleur = "normal" if commune_info['taux_conformite'] > 80 else "inverse" if commune_info['taux_conformite'] < 70 else "off"
            st.metric(
                "Conformité globale",
                f"{commune_info['taux_conformite']:.1f}%",
                f"{random.uniform(-1.5, 2.0):+.1f}%",
                delta_color=conformite_couleur
            )
        
        with col2:
            nb_stations = len(infra['stations'])
            st.metric(
                "Stations d'épuration",
                f"{nb_stations}",
                f"{random.randint(-1, 1)} vs 2023"
            )
        
        with col3:
            rendement_eu = infra['reseaux']['EU']['rendement']
            rendement_couleur = "normal" if rendement_eu > 85 else "inverse"
            st.metric(
                "Rendement réseau EU",
                f"{rendement_eu:.1f}%",
                f"{random.uniform(-0.8, 1.2):+.1f}%",
                delta_color=rendement_couleur
            )
        
        with col4:
            budget = commune_info['budget_assainissement']
            st.metric(
                "Budget annuel",
                f"{budget:.1f} M€",
                f"{random.uniform(-3, 5):+.1f}% vs N-1"
            )
        
        # Deuxième ligne
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Taux de conformité ANC",
                f"{infra['spanc']['taux_conformite_ANC']:.1f}%",
                f"{random.uniform(-2, 1):+.1f}%"
            )
        
        with col2:
            nb_fuites = infra['reseaux']['EU']['nb_fuites_an']
            st.metric(
                "Fuite réseau EU/an",
                f"{nb_fuites}",
                f"{random.randint(-5, 3)} vs 2023"
            )
        
        with col3:
            charge_moyenne = infra['stations']['charge_actuelle'].mean() / infra['stations']['capacite_nominale'].mean() * 100
            st.metric(
                "Charge moyenne STEP",
                f"{charge_moyenne:.1f}%",
                f"{random.uniform(-3, 4):+.1f}%"
            )
        
        with col4:
            ratio_reseau = (commune_info['nb_abonnes_assainissement'] / commune_info['population']) * 100
            st.metric(
                "Taux de couverture",
                f"{ratio_reseau:.1f}%",
                f"{random.uniform(-0.5, 1):+.1f}%"
            )

    def display_stations_epuration(self):
        """Vue détaillée des stations d'épuration"""
        data = self.get_commune_data(st.session_state.selected_commune)
        infra = data['infra']
        
        st.markdown("#### 🏭 Stations de Traitement des Eaux Usées (STEP)")
        
        stations_df = infra['stations'].copy()
        
        # Mise en forme conditionnelle
        def color_conformite(val):
            if val == 'Conforme':
                return 'background-color: #d4edda; color: #155724'
            elif val == 'Non conforme':
                return 'background-color: #f8d7da; color: #721c24'
            else:
                return 'background-color: #fff3cd; color: #856404'
        
        styled_stations = stations_df.style.applymap(color_conformite, subset=['conformite'])
        
        col1, col2 = st.columns([2, 1])
        with col1:
            st.dataframe(styled_stations, use_container_width=True)
        with col2:
            # Graphique de répartition par type
            type_counts = stations_df['type'].value_counts().reset_index()
            type_counts.columns = ['type', 'count']
            fig = px.pie(type_counts, values='count', names='type', 
                        title='Répartition par filière de traitement',
                        color_discrete_sequence=px.colors.qualitative.Set3)
            st.plotly_chart(fig, use_container_width=True)
        
        # Détail des charges
        fig = go.Figure()
        for _, station in stations_df.iterrows():
            fig.add_trace(go.Bar(
                name=station['nom'],
                x=[station['nom']],
                y=[station['charge_actuelle']],
                marker_color='#0066B3',
                text=f"Capacité: {station['capacite_nominale']:.0f} EH<br>Charge: {station['charge_actuelle']/station['capacite_nominale']*100:.1f}%",
                hovertemplate='<b>%{x}</b><br>%{text}<extra></extra>'
            ))
        
        fig.update_layout(
            title="Charge actuelle des STEP (Équivalent-Habitant)",
            xaxis_title="Station",
            yaxis_title="Charge (EH)",
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)

    def display_reseaux_performance(self):
        """Affiche les performances des réseaux d'assainissement"""
        data = self.get_commune_data(st.session_state.selected_commune)
        infra = data['infra']
        reseaux = infra['reseaux']
        
        st.markdown("#### 🕸️ Performance des Réseaux")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Graphique comparatif EU/EP
            df_reseaux = pd.DataFrame([
                {'Réseau': 'Eaux Usées (EU)', 'Longueur (km)': reseaux['EU']['longueur_km'], 
                 'Rendement (%)': reseaux['EU']['rendement'], 'Fuites/an': reseaux['EU']['nb_fuites_an']},
                {'Réseau': 'Eaux Pluviales (EP)', 'Longueur (km)': reseaux['EP']['longueur_km'], 
                 'Rendement (%)': reseaux['EP']['rendement'], 'Fuites/an': reseaux['EP']['nb_fuites_an']}
            ])
            
            fig = px.bar(df_reseaux, x='Réseau', y='Longueur (km)', 
                        title="Longueur du réseau (km)",
                        color='Réseau', 
                        color_discrete_map={'Eaux Usées (EU)': '#0066B3', 'Eaux Pluviales (EP)': '#00A0E2'})
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.bar(df_reseaux, x='Réseau', y='Rendement (%)',
                        title="Rendement du réseau (%)",
                        color='Réseau',
                        color_discrete_map={'Eaux Usées (EU)': '#0066B3', 'Eaux Pluviales (EP)': '#00A0E2'})
            fig.update_layout(yaxis_range=[0, 100])
            st.plotly_chart(fig, use_container_width=True)
        
        # Détail des postes de relevage
        st.markdown("#### ⚙️ Postes de Relevage")
        
        postes_df = infra['postes_relevage'].copy()
        
        # Etat des postes
        etat_counts = postes_df['etat'].value_counts().reset_index()
        etat_counts.columns = ['etat', 'count']
        
        col1, col2 = st.columns([1, 2])
        with col1:
            fig = px.pie(etat_counts, values='count', names='etat',
                        title="État des postes",
                        color='etat',
                        color_discrete_map={'Bon': '#28a745', 'Moyen': '#ffc107', 'Dégradé': '#dc3545'})
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.dataframe(postes_df[['nom', 'debit_m3h', 'etat', 'telemesure', 'dernier_depannage']], 
                        use_container_width=True)

    def display_qualite_eaux(self):
        """Affiche les données historiques de qualité des eaux traitées"""
        data = self.get_commune_data(st.session_state.selected_commune)
        historical_data = data['historical_data']
        
        st.markdown("#### 🧪 Qualité des Eaux Traitées - Évolution")
        
        # Agrégation mensuelle
        monthly_quality = historical_data.groupby('date').agg({
            'db05_mgl': 'mean',
            'dco_mgl': 'mean',
            'mes_mgl': 'mean',
            'ntk_mgl': 'mean',
            'pt_mgl': 'mean',
            'conformite': 'mean'
        }).reset_index()
        monthly_quality['conformite'] = monthly_quality['conformite'] * 100
        
        tab1, tab2 = st.tabs(["📈 Paramètres physico-chimiques", "✅ Taux de conformité"])
        
        with tab1:
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            fig.add_trace(
                go.Scatter(x=monthly_quality['date'], y=monthly_quality['db05_mgl'], 
                          name="DBO5 (mg/L)", line=dict(color='#0066B3', width=2)),
                secondary_y=False,
            )
            fig.add_trace(
                go.Scatter(x=monthly_quality['date'], y=monthly_quality['dco_mgl'], 
                          name="DCO (mg/L)", line=dict(color='#00A0E2', width=2)),
                secondary_y=False,
            )
            fig.add_trace(
                go.Scatter(x=monthly_quality['date'], y=monthly_quality['mes_mgl'], 
                          name="MES (mg/L)", line=dict(color='#0D5332', width=2)),
                secondary_y=False,
            )
            
            fig.update_layout(
                title="Évolution des principaux paramètres de rejet",
                xaxis_title="Date",
                yaxis_title="Concentration (mg/L)",
                hovermode='x unified'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            fig = px.area(monthly_quality, x='date', y='conformite',
                         title="Taux de conformité mensuel des STEP",
                         color_discrete_sequence=['#28a745'])
            fig.add_hline(y=95, line_dash="dash", line_color="green", annotation_text="Objectif 95%")
            fig.add_hline(y=80, line_dash="dash", line_color="orange", annotation_text="Seuil alerte")
            fig.update_layout(yaxis_title="Conformité (%)", yaxis_range=[50, 100])
            st.plotly_chart(fig, use_container_width=True)

    def display_spanc(self):
        """Vue détaillée du Service Public d'Assainissement Non Collectif"""
        data = self.get_commune_data(st.session_state.selected_commune)
        spanc = data['infra']['spanc']
        commune_info = self.communes[st.session_state.selected_commune]
        
        st.markdown("#### 🏠 Assainissement Non Collectif (SPANC)")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Installations ANC", f"{spanc['nb_installations_ANC']:,.0f}")
        with col2:
            st.metric("Taux de conformité", f"{spanc['taux_conformite_ANC']:.1f}%",
                     delta=f"{random.uniform(-1, 2):+.1f}%")
        with col3:
            st.metric("Contrôles/an", f"{spanc['nb_controles_annuels']}")
        
        # Carte des non-conformités (simulation)
        st.markdown("##### Répartition des non-conformités")
        
        non_conformites = {
            'Défaut d\'entretien': random.randint(15, 35),
            'Dispositif vétuste': random.randint(20, 40),
            'Rejet non conforme': random.randint(5, 20),
            'Accessibilité impossible': random.randint(5, 15),
            'Absence de ventilation': random.randint(10, 25),
            'Filière inadaptée': random.randint(10, 30)
        }
        
        df_nc = pd.DataFrame(list(non_conformites.items()), columns=['Cause', 'Nombre'])
        df_nc = df_nc.sort_values('Nombre', ascending=True)
        
        fig = px.bar(df_nc, x='Nombre', y='Cause', 
                    title="Causes des non-conformités ANC",
                    orientation='h',
                    color='Nombre',
                    color_continuous_scale='Reds')
        st.plotly_chart(fig, use_container_width=True)

    def display_travaux_assainissement(self):
        """Affiche le suivi des travaux sur le réseau"""
        data = self.get_commune_data(st.session_state.selected_commune)
        travaux_data = data['travaux_data']
        
        st.markdown('<h3 class="section-header">🚧 TRAVAUX ET RENOUVELLEMENT DU RÉSEAU</h3>', 
                   unsafe_allow_html=True)
        
        if travaux_data.empty:
            st.info("Aucun travaux en cours dans cette commune.")
            return
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # Répartition par statut
            statut_counts = travaux_data['statut'].value_counts().reset_index()
            statut_counts.columns = ['statut', 'count']
            fig = px.pie(statut_counts, values='count', names='statut',
                        title="Statut des chantiers",
                        color='statut',
                        color_discrete_map={'Urgent': '#dc3545', 'Planifié': '#ffc107', 'En cours': '#28a745'})
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Répartition par type
            type_counts = travaux_data['type_travail'].value_counts().reset_index().head(5)
            type_counts.columns = ['type', 'count']
            fig = px.bar(type_counts, x='count', y='type',
                        title="Top 5 des types d'intervention",
                        orientation='h',
                        color='count',
                        color_continuous_scale='Blues')
            st.plotly_chart(fig, use_container_width=True)
        
        # Liste détaillée
        st.markdown("##### 📋 Chantiers en cours et planifiés")
        
        for _, chantier in travaux_data.iterrows():
            urgence_class = "alerte-haute" if chantier['urgence'] > 8 else "alerte-moyenne" if chantier['urgence'] > 5 else "alerte-faible"
            
            with st.container():
                cols = st.columns([1, 2, 1, 1, 1])
                cols[0].markdown(f"**{chantier['id_travail']}**")
                cols[1].markdown(f"{chantier['type_travail']}<br><small>{chantier['localisation']}</small>", 
                                unsafe_allow_html=True)
                cols[2].markdown(f"**{chantier['budget']:.1f} k€**")
                cols[3].markdown(f"{chantier['date_debut']}<br>{chantier['duree_estimee']}j", 
                                unsafe_allow_html=True)
                cols[4].markdown(f"<div class='{urgence_class}'>{chantier['urgence']:.1f}/10</div>", 
                                unsafe_allow_html=True)
                st.markdown("---")

    def display_comparison_assainissement(self):
        """Crée une vue de comparaison entre communes sur l'assainissement"""
        st.markdown('<h3 class="section-header">🏘️ COMPARAISON INTER-COMMUNES - ASSAINISSEMENT</h3>', 
                   unsafe_allow_html=True)
        
        comparison_data = generate_comparison_assainissement(self.communes)
        
        tab1, tab2, tab3 = st.tabs(["Classement conformité", "Taux de couverture", "Performance STEP"])
        
        with tab1:
            col1, col2 = st.columns(2)
            
            with col1:
                top_conformite = comparison_data.sort_values('taux_conformite_global', ascending=False).head(10)
                fig = px.bar(top_conformite, x='nom_complet', y='taux_conformite_global',
                            title="Top 10 - Taux de conformité global",
                            color='taux_conformite_global',
                            color_continuous_scale='RdYlGn')
                fig.update_layout(yaxis_title="Conformité (%)", yaxis_range=[0, 100])
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.scatter(comparison_data, x='population', y='taux_conformite_global',
                                size='budget_assainissement', color='categorie_taille',
                                title="Conformité vs Population",
                                hover_name='nom_complet',
                                size_max=40)
                fig.update_layout(yaxis_title="Conformité (%)")
                st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            fig = px.bar(comparison_data.sort_values('ratio_abonnes_reseau', ascending=False),
                        x='nom_complet', y='ratio_abonnes_reseau',
                        title="Taux de raccordement à l'assainissement collectif",
                        color='ratio_abonnes_reseau',
                        color_continuous_scale='Blues')
            fig.update_layout(yaxis_title="Abonnés / Population (%)", xaxis_tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            col1, col2 = st.columns(2)
            
            with col1:
                top_stations = comparison_data.sort_values('taux_conformite_stations', ascending=False).head(10)
                fig = px.bar(top_stations, x='nom_complet', y='taux_conformite_stations',
                            title="Conformité des stations d'épuration",
                            color='taux_conformite_stations',
                            color_continuous_scale='RdYlGn')
                fig.update_layout(yaxis_title="Stations conformes (%)", yaxis_range=[0, 100])
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.scatter(comparison_data, x='nb_stations', y='rendement_reseau_eu',
                                size='nb_abonnes_assainissement', color='categorie_taille',
                                title="Rendement réseau vs Nombre de STEP",
                                hover_name='nom_complet')
                fig.update_layout(yaxis_title="Rendement réseau EU (%)")
                st.plotly_chart(fig, use_container_width=True)

    def display_strategie_recommandations(self):
        """Affiche les recommandations stratégiques pour l'assainissement"""
        commune_info = self.communes[st.session_state.selected_commune]
        data = self.get_commune_data(st.session_state.selected_commune)
        
        st.markdown('<h3 class="section-header">💡 STRATÉGIE ET PLAN D\'ACTIONS</h3>', 
                   unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### 🎯 Objectifs Réglementaires (DERU)
            
            **Directive Eaux Résiduaires Urbaines:**
            - Conformité ≥ 95% des STEP
            - Rendement épuratoire ≥ 90%
            - Collecte des charges ≥ 98%
            
            **Schéma Directeur Assainissement:**
            1. Renouvellement des réseaux vétustes
            2. Extension des zones de collecte
            3. Mise en séparatif des réseaux unitaires
            4. Gestion des eaux pluviales par temps de pluie
            """)
            
            # Problématiques spécifiques à la commune
            if commune_info['categorie_taille'] in ['micro_commune', 'petite_commune']:
                st.warning("⚠️ **Défis spécifiques**: Zones rurales - priorité au SPANC et à l'extension maîtrisée du réseau.")
            elif commune_info['taux_conformite'] < 75:
                st.error("🚨 **Alerte prioritaire**: Taux de conformité faible. Nécessite un plan de mise en conformité accéléré.")
        
        with col2:
            st.markdown("""
            ### 🚀 Plan d'Investissement 2025-2030
            
            **1. Modernisation des STEP:**
            - Réhabilitation des filières de traitement
            - Optimisation énergétique
            - Traitement du phosphore
            
            **2. Réduction des fuites:**
            - Secteurs de recherche de fuites
            - Renouvellement ciblé des canalisations
            - Objectif rendement cible: 90%
            
            **3. Gestion patrimoniale:**
            - SIG et télégestion
            - Modélisation hydraulique
            - GMAO centralisée
            """)
        
        # Feuille de route
        st.markdown("#### 📅 Feuille de route - Actions prioritaires")
        
        roadmap_data = pd.DataFrame({
            'Action': [
                'Diagnostic réseau EU secteur prioritaire',
                'Mise en conformité STEP', 
                'Renouvellement collecteurs vétustes',
                'Extension réseau zone d\'activité',
                'Campagne de contrôles SPANC',
                'Déploiement télégestion postes'
            ],
            'Échéance': ['T1 2025', 'T2 2025', 'T3 2025', 'T4 2025', '2026', '2027'],
            'Budget (k€)': [120, 850, 650, 430, 90, 280],
            'Priorité': ['Haute', 'Critique', 'Haute', 'Moyenne', 'Moyenne', 'Faible']
        })
        
        def color_priorite(val):
            if val == 'Critique':
                return 'background-color: #dc3545; color: white'
            elif val == 'Haute':
                return 'background-color: #ffc107; color: black'
            elif val == 'Moyenne':
                return 'background-color: #28a745; color: white'
            else:
                return 'background-color: #6c757d; color: white'
        
        st.dataframe(roadmap_data.style.applymap(color_priorite, subset=['Priorité']), 
                    use_container_width=True)

    def run(self):
        """Fonction principale pour exécuter le dashboard"""
        self.display_commune_selector()
        self.display_header()
        self.display_key_metrics()
        
        # Mise à jour automatique
        if st.sidebar.button("🔄 Mettre à jour les données"):
            self.update_live_data(st.session_state.selected_commune)
            st.success("✅ Données assainissement mises à jour!")
        
        # Sidebar - infos et alertes
        commune_info = self.communes[st.session_state.selected_commune]
        with st.sidebar:
            st.markdown("### 📍 Point de vigilance")
            
            data = self.get_commune_data(st.session_state.selected_commune)
            infra = data['infra']
            
            # Alertes automatiques
            if commune_info['taux_conformite'] < 75:
                st.error("⚠️ Taux de conformité global critique")
            if infra['reseaux']['EU']['rendement'] < 75:
                st.error("🕸️ Rendement réseau EU très faible")
            if infra['spanc']['taux_conformite_ANC'] < 65:
                st.warning("🏠 Fort taux de non-conformité ANC")
            
            # Filtre par catégorie
            st.markdown("### 🔍 Filtres comparaison")
            st.selectbox(
                "Voir communes par taille:",
                ['Toutes', 'Villes principales', 'Grandes communes', 
                 'Communes moyennes', 'Petites communes', 'Micro-communes']
            )
        
        # Création des onglets
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🏭 Stations & Réseaux", 
            "🧪 Qualité & Conformité",
            "🏠 SPANC", 
            "🚧 Travaux",
            "📊 Comparatif & Stratégie"
        ])
        
        with tab1:
            self.display_stations_epuration()
            self.display_reseaux_performance()
        
        with tab2:
            self.display_qualite_eaux()
        
        with tab3:
            self.display_spanc()
        
        with tab4:
            self.display_travaux_assainissement()
        
        with tab5:
            self.display_comparison_assainissement()
            self.display_strategie_recommandations()
        
        # Footer
        st.markdown("---")
        st.markdown("""
        **Dashboard Assainissement Île de La Réunion** | Données mises à jour en temps réel |
        **24 communes couvertes** | Sources: Office de l'Eau Réunion, ARS, DEAL, SPANC |
        © 2024 Conseil Départemental de La Réunion
        """)

if __name__ == "__main__":
    dashboard = AssainissementReunionDashboard()
    dashboard.run()
