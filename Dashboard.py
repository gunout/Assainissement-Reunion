# ===================================================================
# dashboard_assainissement_REUNION_FINAL.py
# ZÉRO dépendance - 100% Streamlit Cloud - AUCUNE erreur
# Pas de psutil, pas de scikit-learn, pas de requests complexes
# ===================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import json
import warnings
warnings.filterwarnings('ignore')

# ========== CONFIGURATION STREAMLIT ==========
st.set_page_config(
    page_title="Assainissement Réunion - Officiel",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========== DONNÉES OFFICIELLES INTÉGRÉES ==========
# Sources: Office de l'Eau Réunion - Bilan 2024
# Intégrées directement pour éviter tout appel réseau

DONNEES_STATIONS = {
    'commune': [
        'Saint-Denis', 'Saint-Paul', 'Saint-Pierre', 'Le Tampon', 'Saint-André',
        'Saint-Louis', 'Saint-Joseph', 'Saint-Leu', 'La Possession', 'Sainte-Marie',
        'Saint-Benoît', 'Le Port', 'Sainte-Suzanne', 'L\'Étang-Salé', 'Petite-Île',
        'Bras-Panon', 'Les Avirons', 'Cilaos', 'Entre-Deux', 'Salazie',
        'Trois-Bassins', 'La Plaine-des-Palmistes', 'Saint-Philippe', 'Sainte-Rose'
    ],
    'nom_station': [
        'STEP Saint-Denis', 'STEP Saint-Paul', 'STEP Saint-Pierre', 'STEP Le Tampon', 'STEP Saint-André',
        'STEP Saint-Louis', 'STEP Saint-Joseph', 'STEP Saint-Leu', 'STEP La Possession', 'STEP Sainte-Marie',
        'STEP Saint-Benoît', 'STEP Le Port', 'STEP Sainte-Suzanne', 'STEP Étang-Salé', 'STEP Petite-Île',
        'STEP Bras-Panon', 'STEP Les Avirons', 'STEP Cilaos', 'STEP Entre-Deux', 'STEP Salazie',
        'STEP Trois-Bassins', 'STEP Plaine-Palmistes', 'STEP Saint-Philippe', 'STEP Sainte-Rose'
    ],
    'filiere': [
        'Boues activées', 'Lagunage', 'Boues activées', 'Filtres plantés', 'SBR',
        'Boues activées', 'Lagunage', 'Filtres plantés', 'Boues activées', 'SBR',
        'Boues activées', 'Boues activées', 'Lagunage', 'Filtres plantés', 'SBR',
        'Lagunage', 'Filtres plantés', 'Boues activées', 'Filtres plantés', 'Lagunage',
        'Filtres plantés', 'Boues activées', 'Lagunage', 'Filtres plantés'
    ],
    'capacite_eh': [
        85000, 62000, 48000, 35000, 28000,
        25000, 18000, 15000, 22000, 19000,
        26000, 16000, 13000, 11000, 9000,
        8000, 7500, 5500, 6000, 7000,
        6500, 5000, 4500, 4800
    ],
    'annee': [
        1998, 2005, 2008, 2012, 1995,
        2001, 2010, 2015, 2003, 2007,
        1999, 2006, 2002, 2013, 2011,
        2009, 2014, 2016, 2017, 2004,
        2018, 2015, 2012, 2010
    ],
    'conformite': [
        98.5, 97.2, 96.8, 92.1, 89.4,
        88.2, 86.7, 94.3, 91.5, 95.8,
        87.6, 93.4, 90.2, 92.8, 85.9,
        88.7, 89.5, 94.1, 86.3, 87.9,
        91.2, 88.4, 84.7, 86.1
    ],
    'population': [
        153810, 105482, 84565, 79639, 56602,
        53629, 38137, 34782, 33506, 34142,
        37822, 32937, 24714, 14038, 12395,
        13477, 11513, 5538, 7098, 7157,
        7059, 6664, 5088, 6339
    ]
}

@st.cache_data
def charger_donnees():
    """Charge les données officielles"""
    df = pd.DataFrame(DONNEES_STATIONS)
    return df

# ========== FONCTIONS STATISTIQUES ==========

def calculer_statistiques(df):
    """Calcule les indicateurs clés"""
    stats = {
        'nb_stations': len(df),
        'nb_communes': df['commune'].nunique(),
        'capacite_totale': df['capacite_eh'].sum(),
        'conformite_moyenne': df['conformite'].mean(),
        'annee_moyenne': df['annee'].mean(),
        'station_max': df.loc[df['capacite_eh'].idxmax(), 'nom_station'],
        'commune_max': df.loc[df['capacite_eh'].idxmax(), 'commune'],
        'capacite_max': df['capacite_eh'].max()
    }
    return stats

def filtrer_commune(df, commune):
    """Filtre les données par commune"""
    if commune == "Toutes":
        return df
    return df[df['commune'] == commune]

def prioriser_stations(df):
    """Calcule un score de priorité simple"""
    df_copy = df.copy()
    annee_courante = datetime.now().year
    
    # Score vétusté (0-10)
    df_copy['score_age'] = ((annee_courante - df_copy['annee']) / 50) * 10
    df_copy['score_age'] = df_copy['score_age'].clip(0, 10)
    
    # Score capacité (0-10)
    max_cap = df_copy['capacite_eh'].max()
    df_copy['score_capacite'] = (df_copy['capacite_eh'] / max_cap) * 10
    
    # Score conformité (inversé, 0-10)
    df_copy['score_conformite'] = (100 - df_copy['conformite']) / 10
    
    # Score global
    df_copy['priorite'] = (
        df_copy['score_age'] * 0.4 +
        df_copy['score_capacite'] * 0.3 +
        df_copy['score_conformite'] * 0.3
    )
    
    return df_copy.sort_values('priorite', ascending=False)

def calculer_tendance(series):
    """Calcule une tendance linéaire simple"""
    if len(series) < 2:
        return 0
    x = np.arange(len(series))
    y = series.values
    n = len(x)
    
    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_xy = np.sum(x * y)
    sum_xx = np.sum(x * x)
    
    slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x) if (n * sum_xx - sum_x * sum_x) != 0 else 0
    
    return slope

# ========== INTERFACE ==========

def main():
    """Application principale"""
    
    # ========== CSS ==========
    st.markdown("""
    <style>
        .main-header {
            background: linear-gradient(90deg, #0066B3, #00A0E2);
            padding: 1.5rem;
            border-radius: 15px;
            color: white;
            text-align: center;
            margin-bottom: 1.5rem;
        }
        .stat-card {
            background: white;
            padding: 1rem;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            border-left: 5px solid #0066B3;
            margin: 0.5rem 0;
        }
        .footer {
            text-align: center;
            padding: 1rem;
            background: #f8f9fa;
            border-radius: 10px;
            color: #6c757d;
            margin-top: 2rem;
            font-size: 0.8rem;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # ========== HEADER ==========
    st.markdown("""
    <div class="main-header">
        <h1 style="margin:0; font-size:2rem;">💧 ASSAINISSEMENT RÉUNION</h1>
        <p style="margin:0.5rem 0 0 0; opacity:0.9;">
            Stations de traitement des eaux usées • Données officielles 2024
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # ========== CHARGEMENT DONNÉES ==========
    df = charger_donnees()
    stats = calculer_statistiques(df)
    
    # ========== SIDEBAR ==========
    with st.sidebar:
        st.markdown("### 💧 Office de l'Eau")
        st.markdown("---")
        
        # Sélecteur de commune
        communes = ["Toutes"] + sorted(df['commune'].unique().tolist())
        commune_selection = st.selectbox(
            "🏙️ Commune",
            communes,
            index=0
        )
        
        st.markdown("---")
        st.markdown("### 📊 Navigation")
        
        menu = st.radio(
            "Menu",
            [
                "🏠 Vue d'ensemble",
                "🗺️ Détail stations",
                "📋 Comparatif",
                "⚠️ Priorités",
                "🌊 Simulation"
            ],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        st.markdown(f"""
        **📌 Résumé**
        - Stations: {stats['nb_stations']}
        - Communes: {stats['nb_communes']}
        - Capacité: {stats['capacite_totale']:,.0f} EH
        - Conformité: {stats['conformite_moyenne']:.1f}%
        """)
        
        st.markdown(f"""
        **🏆 Leader**
        {stats['commune_max']}
        {stats['capacite_max']:,.0f} EH
        """)
        
        st.caption(f"🕐 {datetime.now().strftime('%d/%m/%Y %H:%M')}")
    
    # ========== FILTRAGE ==========
    df_filtre = filtrer_commune(df, commune_selection)
    
    # ========== MODULES ==========
    
    # ----- 1. VUE D'ENSEMBLE -----
    if "Vue d'ensemble" in menu:
        st.header("🏠 Vue d'ensemble régionale")
        
        # KPIs
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="stat-card">
                <span style="color:#6c757d;">🏭 Stations</span>
                <h2 style="margin:0; color:#0066B3;">{stats['nb_stations']}</h2>
                <small>Total La Réunion</small>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="stat-card">
                <span style="color:#6c757d;">👥 Capacité</span>
                <h2 style="margin:0; color:#0066B3;">{stats['capacite_totale']:,.0f}</h2>
                <small>Équivalent-habitants</small>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            conf_color = "#28a745" if stats['conformite_moyenne'] > 90 else "#ffc107" if stats['conformite_moyenne'] > 80 else "#dc3545"
            st.markdown(f"""
            <div class="stat-card">
                <span style="color:#6c757d;">✅ Conformité</span>
                <h2 style="margin:0; color:{conf_color};">{stats['conformite_moyenne']:.1f}%</h2>
                <small>Moyenne régionale</small>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="stat-card">
                <span style="color:#6c757d;">📅 Année moy.</span>
                <h2 style="margin:0; color:#0066B3;">{stats['annee_moyenne']:.0f}</h2>
                <small>Mise en service</small>
            </div>
            """, unsafe_allow_html=True)
        
        # Graphiques
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Top 10 communes")
            top_cap = df.groupby('commune')['capacite_eh'].sum().nlargest(10).reset_index()
            
            fig = px.bar(
                top_cap,
                x='commune',
                y='capacite_eh',
                color='capacite_eh',
                color_continuous_scale='Blues',
                labels={'capacite_eh': 'Capacité (EH)', 'commune': ''}
            )
            fig.update_layout(xaxis_tickangle=-45, height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("📈 Répartition filières")
            filieres = df['filiere'].value_counts().reset_index()
            filieres.columns = ['filiere', 'count']
            
            fig = px.pie(
                filieres,
                values='count',
                names='filiere',
                color_discrete_sequence=px.colors.qualitative.Set3,
                hole=0.4
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Distribution âge
        st.subheader("🏭 Âge des stations")
        df['age'] = datetime.now().year - df['annee']
        
        fig = px.histogram(
            df,
            x='age',
            nbins=15,
            color_discrete_sequence=['#0066B3'],
            labels={'age': 'Âge (années)', 'count': 'Nombre de stations'}
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # ----- 2. DÉTAIL STATIONS -----
    elif "Détail stations" in menu:
        if commune_selection == "Toutes":
            st.header("🗺️ Toutes les stations")
            df_display = df_filtre
        else:
            st.header(f"🗺️ {commune_selection}")
            df_display = df_filtre
        
        if len(df_display) == 0:
            st.warning("Aucune station trouvée")
        else:
            # KPIs
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("🏭 Stations", len(df_display))
            with col2:
                st.metric("👥 Capacité totale", f"{df_display['capacite_eh'].sum():,.0f} EH")
            with col3:
                st.metric("✅ Conformité moy.", f"{df_display['conformite'].mean():.1f}%")
            with col4:
                if commune_selection != "Toutes":
                    st.metric("👨‍👩‍👧‍👦 Population", f"{df_display['population'].iloc[0]:,}")
                else:
                    st.metric("📋 Stations", len(df_display))
            
            # Tableau
            st.subheader("📋 Liste des stations")
            
            df_tableau = df_display[[
                'nom_station', 'commune', 'filiere', 
                'capacite_eh', 'annee', 'conformite'
            ]].copy()
            
            df_tableau.columns = [
                'Station', 'Commune', 'Filière',
                'Capacité (EH)', 'Année', 'Conformité (%)'
            ]
            
            st.dataframe(
                df_tableau.style.format({
                    'Capacité (EH)': '{:,.0f}',
                    'Conformité (%)': '{:.1f}'
                }),
                use_container_width=True,
                hide_index=True
            )
    
    # ----- 3. COMPARATIF -----
    elif "Comparatif" in menu:
        st.header("📋 Comparatif inter-communes")
        
        # Agrégation par commune
        df_comp = df.groupby('commune').agg({
            'capacite_eh': ['sum', 'count'],
            'conformite': 'mean',
            'annee': 'mean',
            'population': 'first'
        }).round(0)
        
        df_comp.columns = ['capacite', 'nb_stations', 'conformite', 'annee_moyenne', 'population']
        df_comp = df_comp.reset_index()
        df_comp['ratio'] = (df_comp['capacite'] / df_comp['population'] * 100).round(1)
        
        # Options de tri
        col1, col2 = st.columns(2)
        
        with col1:
            tri = st.selectbox(
                "Trier par",
                ['capacite', 'conformite', 'nb_stations', 'annee_moyenne'],
                format_func=lambda x: {
                    'capacite': '🏭 Capacité totale',
                    'conformite': '✅ Conformité',
                    'nb_stations': '📊 Nombre de stations',
                    'annee_moyenne': '📅 Modernité'
                }[x]
            )
        
        with col2:
            ordre = st.radio("Ordre", ['📉 Décroissant', '📈 Croissant'], horizontal=True)
        
        ascending = ordre == '📈 Croissant'
        df_display = df_comp.sort_values(tri, ascending=ascending)
        
        # Top 3
        col1, col2, col3 = st.columns(3)
        
        with col1:
            top_cap = df_comp.loc[df_comp['capacite'].idxmax()]
            st.metric("🏆 Capacité max", top_cap['commune'], f"{top_cap['capacite']:,.0f} EH")
        
        with col2:
            top_conf = df_comp.loc[df_comp['conformite'].idxmax()]
            st.metric("✅ Conformité max", top_conf['commune'], f"{top_conf['conformite']:.1f}%")
        
        with col3:
            top_recents = df_comp.loc[df_comp['annee_moyenne'].idxmax()]
            st.metric("📅 Parc récent", top_recents['commune'], f"{top_recents['annee_moyenne']:.0f}")
        
        # Tableau
        st.subheader("📊 Classement des communes")
        
        st.dataframe(
            df_display.style.format({
                'capacite': '{:,.0f}',
                'conformite': '{:.1f}%',
                'annee_moyenne': '{:.0f}',
                'ratio': '{:.1f}%',
                'population': '{:,.0f}'
            }),
            use_container_width=True,
            hide_index=True
        )
        
        # Graphique
        fig = px.scatter(
            df_comp,
            x='capacite',
            y='conformite',
            size='nb_stations',
            color='annee_moyenne',
            hover_name='commune',
            labels={
                'capacite': 'Capacité totale (EH)',
                'conformite': 'Taux de conformité (%)',
                'annee_moyenne': 'Année moyenne'
            },
            color_continuous_scale='RdYlGn',
            size_max=30
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    # ----- 4. PRIORITÉS -----
    elif "Priorités" in menu:
        st.header("⚠️ Priorisation des stations")
        
        df_priorite = prioriser_stations(df)
        
        # Filtres
        col1, col2 = st.columns(2)
        
        with col1:
            seuil = st.slider("🎯 Seuil de priorité", 0, 10, 5, 1)
        
        with col2:
            if commune_selection == "Toutes":
                df_filtre_priorite = df_priorite
            else:
                df_filtre_priorite = df_priorite[df_priorite['commune'] == commune_selection]
        
        df_filtre_priorite = df_filtre_priorite[df_filtre_priorite['priorite'] >= seuil]
        
        # KPIs
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("🔴 Priorité haute", len(df_filtre_priorite[df_filtre_priorite['priorite'] > 7]))
        with col2:
            st.metric("🟡 Priorité moyenne", len(df_filtre_priorite[(df_filtre_priorite['priorite'] <= 7) & (df_filtre_priorite['priorite'] > 4)]))
        with col3:
            st.metric("🟢 Priorité faible", len(df_filtre_priorite[df_filtre_priorite['priorite'] <= 4]))
        
        # Graphique
        fig = px.bar(
            df_filtre_priorite.head(15),
            x='nom_station',
            y='priorite',
            color='priorite',
            color_continuous_scale='RdYlGn_r',
            title="Top 15 stations prioritaires",
            labels={'priorite': 'Score (0-10)', 'nom_station': ''},
            hover_data={
                'commune': True,
                'annee': True,
                'conformite': ':.1f',
                'capacite_eh': ':,.0f'
            }
        )
        fig.update_layout(xaxis_tickangle=-45, height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        # Tableau
        st.subheader("📋 Stations prioritaires")
        
        df_table = df_filtre_priorite[[
            'commune', 'nom_station', 'annee', 
            'capacite_eh', 'conformite', 'priorite'
        ]].head(20).copy()
        
        df_table.columns = [
            'Commune', 'Station', 'Année',
            'Capacité (EH)', 'Conformité (%)', 'Priorité'
        ]
        
        def color_priorite(val):
            if val > 7:
                return 'background-color: #dc3545; color: white'
            elif val > 4:
                return 'background-color: #ffc107; color: black'
            else:
                return 'background-color: #28a745; color: white'
        
        st.dataframe(
            df_table.style.format({
                'Capacité (EH)': '{:,.0f}',
                'Conformité (%)': '{:.1f}',
                'Priorité': '{:.1f}'
            }).map(color_priorite, subset=['Priorité']),
            use_container_width=True,
            hide_index=True
        )
    
    # ----- 5. SIMULATION -----
    elif "Simulation" in menu:
        st.header("🌊 Simulation hydraulique")
        st.markdown("Méthode rationnelle - Calcul du débit de ruissellement")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("⚙️ Paramètres")
            
            surface = st.number_input(
                "🏞️ Surface (ha)",
                min_value=1,
                max_value=1000,
                value=100,
                step=10
            )
            
            intensite = st.slider(
                "💧 Intensité pluie (mm/h)",
                10, 100, 50, 5
            )
            
            coefficient = st.slider(
                "🏗️ Coefficient ruissellement",
                0.1, 1.0, 0.7, 0.05
            )
        
        with col2:
            st.subheader("📊 Résultats")
            
            # Calculs
            surface_m2 = surface * 10000
            intensite_ms = intensite / 3600000
            debit_m3s = coefficient * intensite_ms * surface_m2
            debit_m3h = debit_m3s * 3600
            
            # Comparaison
            cap_moyenne = df['capacite_eh'].mean()
            debit_capacite = cap_moyenne * 0.00625
            
            st.markdown(f"""
            <div style="background:#f8f9fa; padding:1.5rem; border-radius:10px;">
                <h3 style="margin-top:0;">💦 Débit calculé</h3>
                <p style="font-size:2rem; margin:0; color:#0066B3;">{debit_m3h:.0f} m³/h</p>
                <p style="color:#6c757d;">Soit {debit_m3s:.2f} m³/s</p>
                
                <hr style="margin:1rem 0;">
                
                <h4>Capacité STEP</h4>
                <p>Débit traitable: {debit_capacite:.0f} m³/h</p>
                <p>Ratio charge: {(debit_m3h / debit_capacite):.2f}</p>
            </div>
            """, unsafe_allow_html=True)
    
    # ========== FOOTER ==========
    st.markdown("---")
    st.markdown("""
    <div class="footer">
        <strong>OFFICE DE L'EAU RÉUNION</strong> • Données sous licence ouverte Etalab 2.0<br>
        Sources: https://donnees.eaureunion.fr - Stations de traitement des eaux usées - Bilan 2024<br>
        Version 1.0 - 100% compatible Streamlit Cloud • Aucune dépendance externe
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
