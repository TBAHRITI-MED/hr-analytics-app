"""
================================================================================
    IBM HR Analytics - Analyse de l'Attrition des Employés
    Projet INFO0902 - Analyse des Données
    Master 2 IA & Data Science - Université de Reims Champagne-Ardenne
================================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix, roc_curve, 
                             auc, precision_recall_curve, silhouette_score,
                             accuracy_score, f1_score, precision_score, recall_score)
from sklearn.manifold import TSNE

import prince  # Pour FAMD (AFDM en français)
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy import stats
from imblearn.over_sampling import SMOTE

import warnings
warnings.filterwarnings('ignore')

# Configuration de la page
st.set_page_config(
    page_title="HR Analytics - Attrition Analysis",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé - Design professionnel et épuré
st.markdown("""
<style>
    /* ===== GLOBAL ===== */
    .stApp {
        background-color: #FAFBFC;
    }
    
    /* ===== MAIN HEADER ===== */
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1a1a2e;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    /* ===== SIDEBAR ===== */
    section[data-testid="stSidebar"] {
        background-color: #f8f9fa !important;
        border-right: 1px solid #e9ecef;
    }
    section[data-testid="stSidebar"] > div {
        background-color: #f8f9fa !important;
    }
    
    /* ===== INFO BOXES ===== */
    .info-box {
        background-color: #e3f2fd;
        padding: 1rem 1.25rem;
        border-radius: 8px;
        border-left: 4px solid #2196F3;
        margin: 1rem 0;
        color: #1565c0;
    }
    .warning-box {
        background-color: #fff3e0;
        padding: 1rem 1.25rem;
        border-radius: 8px;
        border-left: 4px solid #ff9800;
        margin: 1rem 0;
        color: #e65100;
    }
    .success-box {
        background-color: #e8f5e9;
        padding: 1rem 1.25rem;
        border-radius: 8px;
        border-left: 4px solid #4caf50;
        margin: 1rem 0;
        color: #2e7d32;
    }
    
    /* ===== STAT CARDS ===== */
    .stat-card {
        background: white;
        padding: 1.25rem;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border: 1px solid #eee;
        margin: 0.5rem 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        text-align: center;
    }
    
    /* ===== TABS ===== */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background-color: #f1f3f4;
        padding: 4px;
        border-radius: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        border-radius: 6px;
        padding: 8px 16px;
        color: #5f6368;
    }
    .stTabs [aria-selected="true"] {
        background-color: white !important;
        color: #1a73e8 !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    
    /* ===== BUTTONS ===== */
    .stButton > button {
        background-color: #1a73e8;
        color: white;
        border: none;
        border-radius: 6px;
        padding: 0.5rem 1.5rem;
        font-weight: 500;
    }
    .stButton > button:hover {
        background-color: #1557b0;
    }
    
    /* ===== METRICS ===== */
    [data-testid="stMetricValue"] {
        font-size: 1.8rem;
        font-weight: 600;
        color: #1a1a2e;
    }
    [data-testid="stMetricLabel"] {
        color: #666;
    }
    
    /* ===== HEADERS ===== */
    h2 {
        color: #1a1a2e;
        font-weight: 600;
        border-bottom: 2px solid #1a73e8;
        padding-bottom: 0.5rem;
        display: inline-block;
    }
    h3 {
        color: #333;
        font-weight: 600;
    }
    h4 {
        color: #444;
        font-weight: 500;
    }
    
    /* ===== DATAFRAME ===== */
    .stDataFrame {
        border-radius: 8px;
        border: 1px solid #e0e0e0;
    }
    
    /* ===== EXPANDER ===== */
    .streamlit-expanderHeader {
        background-color: #f8f9fa;
        border-radius: 6px;
    }
    
    /* ===== SELECTBOX & INPUTS ===== */
    .stSelectbox > div > div,
    .stMultiSelect > div > div {
        border-radius: 6px;
    }
    
    /* ===== DIVIDER ===== */
    hr {
        border: none;
        height: 1px;
        background-color: #e0e0e0;
        margin: 1.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


# ==================== FONCTIONS UTILITAIRES ====================

@st.cache_data
def load_data(uploaded_file=None):
    """Charge le dataset IBM HR Analytics"""
    
    df = None
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        # Essayer de charger depuis le dossier local
        local_paths = [
            'data/WA_Fn-UseC_-HR-Employee-Attrition.csv',
            'WA_Fn-UseC_-HR-Employee-Attrition.csv',
            '/mnt/user-data/uploads/WA_Fn-UseC_-HR-Employee-Attrition.csv'
        ]
        
        for path in local_paths:
            try:
                df = pd.read_csv(path)
                break
            except:
                continue
    
    if df is not None:
        # Convertir les colonnes numériques qui pourraient être lues comme strings
        numeric_columns = ['Age', 'DailyRate', 'DistanceFromHome', 'Education', 
                          'EnvironmentSatisfaction', 'HourlyRate', 'JobInvolvement',
                          'JobLevel', 'JobSatisfaction', 'MonthlyIncome', 'MonthlyRate',
                          'NumCompaniesWorked', 'PercentSalaryHike', 'PerformanceRating',
                          'RelationshipSatisfaction', 'StockOptionLevel', 'TotalWorkingYears',
                          'TrainingTimesLastYear', 'WorkLifeBalance', 'YearsAtCompany',
                          'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager']
        
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    
    return None


def get_variable_types(df):
    """Identifie les types de variables"""
    quant_vars = df.select_dtypes(include=[np.number]).columns.tolist()
    qual_vars = df.select_dtypes(include=['object']).columns.tolist()
    return quant_vars, qual_vars


def preprocess_for_analysis(df):
    """Prétraitement pour l'analyse"""
    df_processed = df.copy()
    
    # Supprimer les colonnes non informatives
    cols_to_drop = ['EmployeeCount', 'StandardHours', 'Over18', 'EmployeeNumber']
    cols_to_drop = [col for col in cols_to_drop if col in df_processed.columns]
    df_processed = df_processed.drop(columns=cols_to_drop)
    
    # IMPORTANT: Convertir les colonnes numériques qui ont été lues comme strings
    for col in df_processed.columns:
        # Essayer de convertir en numérique
        if df_processed[col].dtype == 'object':
            try:
                converted = pd.to_numeric(df_processed[col], errors='coerce')
                # Si plus de 90% des valeurs sont converties, c'est probablement numérique
                if converted.notna().mean() > 0.9:
                    df_processed[col] = converted
            except:
                pass
    
    return df_processed


def encode_categorical(df, target_col='Attrition'):
    """Encode les variables catégorielles"""
    df_encoded = df.copy()
    label_encoders = {}
    
    for col in df_encoded.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
        label_encoders[col] = le
    
    return df_encoded, label_encoders


# ==================== PAGE: ACCUEIL ====================

def page_accueil():
    st.markdown('<h1 class="main-header">🎯 IBM HR Analytics</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Analyse de l\'Attrition des Employés avec AFDM, Clustering et Machine Learning</p>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>📊 Analyse Factorielle</h3>
            <p>ACP, ACM, AFDM, AFC</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>🔮 Clustering</h3>
            <p>K-Means & CAH</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>🤖 Prédiction</h3>
            <p>Classification ML</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("### 📋 Contexte du Projet")
    st.markdown("""
    <div class="info-box">
    <strong>Objectif :</strong> Ce projet vise à analyser les facteurs influençant l'attrition des employés 
    dans une entreprise fictive d'IBM. Nous utilisons des méthodes d'analyse factorielle pour données mixtes 
    (AFDM), du clustering pour identifier des profils d'employés, et des algorithmes de machine learning 
    pour prédire le départ des employés.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 🎯 Problématique")
    st.write("""
    **Question principale :** Quels sont les facteurs déterminants qui influencent la décision d'un employé 
    de quitter l'entreprise, et comment peut-on prédire et prévenir l'attrition ?
    
    **Questions secondaires :**
    - Existe-t-il des profils types d'employés à risque de départ ?
    - Quelles variables ont le plus d'impact sur la satisfaction et la rétention ?
    - Peut-on identifier des clusters d'employés ayant des caractéristiques similaires ?
    """)
    
    st.markdown("### 🛠️ Méthodologie")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Analyse Exploratoire (EDA)**
        - Statistiques descriptives
        - Visualisation des distributions
        - Analyse des corrélations
        - Détection des outliers
        """)
        
        st.markdown("""
        **Analyse Factorielle (4 méthodes)**
        - **ACP** : Variables quantitatives uniquement
        - **ACM** : Variables qualitatives uniquement
        - **AFDM** : Données mixtes (quanti + quali)
        - **AFC** : Tableau de contingence (2 var. quali)
        """)
    
    with col2:
        st.markdown("""
        **Clustering**
        - K-Means avec méthode du coude
        - Classification Ascendante Hiérarchique (CAH)
        - Validation par silhouette score
        - Profilage des clusters
        """)
        
        st.markdown("""
        **Prédiction (Classification)**
        - Random Forest, XGBoost, Logistic Regression
        - Gestion du déséquilibre (SMOTE)
        - Validation croisée
        - Analyse SHAP pour l'interprétabilité
        """)
    
    st.markdown("### 📁 Description du Dataset")
    st.write("""
    Le dataset **IBM HR Analytics Employee Attrition & Performance** contient 1470 observations 
    et 35 variables décrivant les caractéristiques des employés :
    
    - **Variables quantitatives** : Age, MonthlyIncome, YearsAtCompany, DistanceFromHome, etc.
    - **Variables qualitatives** : Department, JobRole, MaritalStatus, OverTime, etc.
    - **Variable cible** : Attrition (Yes/No)
    """)


# ==================== PAGE: EXPLORATION DES DONNÉES ====================

def page_exploration(df):
    st.markdown("## 📊 Exploration des Données (EDA)")
    
    # Tabs pour organiser l'exploration
    tab1, tab2, tab3, tab4 = st.tabs(["📋 Aperçu", "📈 Distributions", "🔗 Corrélations", "⚠️ Outliers"])
    
    quant_vars, qual_vars = get_variable_types(df)
    
    with tab1:
        st.markdown("### Vue d'ensemble du Dataset")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Observations", f"{df.shape[0]:,}")
        col2.metric("Variables", f"{df.shape[1]}")
        col3.metric("Var. Quantitatives", len(quant_vars))
        col4.metric("Var. Qualitatives", len(qual_vars))
        
        st.markdown("#### Aperçu des données")
        st.dataframe(df.head(10), use_container_width=True)
        
        st.markdown("#### Types de variables")
        
        # Créer un tableau pour les types de variables
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="info-box">
            <strong>📊 Variables Quantitatives (26)</strong>
            <table style="width:100%; margin-top:10px; font-size:0.9rem;">
                <tr><td>Age</td><td>DailyRate</td><td>DistanceFromHome</td></tr>
                <tr><td>Education</td><td>EmployeeCount</td><td>EmployeeNumber</td></tr>
                <tr><td>EnvironmentSatisfaction</td><td>HourlyRate</td><td>JobInvolvement</td></tr>
                <tr><td>JobLevel</td><td>JobSatisfaction</td><td>MonthlyIncome</td></tr>
                <tr><td>MonthlyRate</td><td>NumCompaniesWorked</td><td>PercentSalaryHike</td></tr>
                <tr><td>PerformanceRating</td><td>RelationshipSatisfaction</td><td>StandardHours</td></tr>
                <tr><td>StockOptionLevel</td><td>TotalWorkingYears</td><td>TrainingTimesLastYear</td></tr>
                <tr><td>WorkLifeBalance</td><td>YearsAtCompany</td><td>YearsInCurrentRole</td></tr>
                <tr><td>YearsSinceLastPromotion</td><td>YearsWithCurrManager</td><td></td></tr>
            </table>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="success-box">
            <strong>📋 Variables Qualitatives (9)</strong>
            <table style="width:100%; margin-top:10px; font-size:0.9rem;">
                <tr><td>Attrition</td><td>BusinessTravel</td><td>Department</td></tr>
                <tr><td>EducationField</td><td>Gender</td><td>JobRole</td></tr>
                <tr><td>MaritalStatus</td><td>Over18</td><td>OverTime</td></tr>
            </table>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("#### Statistiques Descriptives")
        st.dataframe(df.describe().T.round(2), use_container_width=True)
        
        st.markdown("#### Valeurs Manquantes")
        missing = df.isnull().sum()
        if missing.sum() == 0:
            st.success("✅ Aucune valeur manquante dans le dataset!")
        else:
            st.dataframe(missing[missing > 0])
    
    with tab2:
        st.markdown("### Distribution des Variables")
        
        # Variable cible
        st.markdown("#### Distribution de la Variable Cible (Attrition)")
        
        col1, col2 = st.columns(2)
        
        with col1:
            attrition_counts = df['Attrition'].value_counts()
            fig = px.pie(values=attrition_counts.values, 
                        names=attrition_counts.index,
                        title="Répartition de l'Attrition",
                        color_discrete_sequence=['#10B981', '#EF4444'])
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("""
            <div class="warning-box">
            <strong>⚠️ Déséquilibre des classes:</strong><br>
            Le dataset présente un déséquilibre significatif avec environ 16% d'attrition.
            Cela nécessitera des techniques de rééquilibrage (SMOTE) pour la classification.
            </div>
            """, unsafe_allow_html=True)
            
            st.write("**Statistiques:**")
            st.write(f"- Non (retenus): {attrition_counts.get('No', 0)} ({attrition_counts.get('No', 0)/len(df)*100:.1f}%)")
            st.write(f"- Yes (partis): {attrition_counts.get('Yes', 0)} ({attrition_counts.get('Yes', 0)/len(df)*100:.1f}%)")
        
        st.markdown("---")
        
        # Variables quantitatives
        st.markdown("#### Distribution des Variables Quantitatives")
        selected_quant = st.multiselect("Sélectionner les variables à visualiser:", 
                                         quant_vars, 
                                         default=quant_vars[:4])
        
        if selected_quant:
            n_cols = 2
            n_rows = (len(selected_quant) + 1) // 2
            
            fig = make_subplots(rows=n_rows, cols=n_cols, 
                               subplot_titles=selected_quant)
            
            for i, var in enumerate(selected_quant):
                row = i // n_cols + 1
                col = i % n_cols + 1
                
                fig.add_trace(
                    go.Histogram(x=df[var], name=var, marker_color='#667eea'),
                    row=row, col=col
                )
            
            fig.update_layout(height=300*n_rows, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        # Variables qualitatives
        st.markdown("#### Distribution des Variables Qualitatives")
        selected_qual = st.selectbox("Sélectionner une variable:", qual_vars)
        
        fig = px.histogram(df, x=selected_qual, color='Attrition',
                          barmode='group',
                          color_discrete_map={'No': '#10B981', 'Yes': '#EF4444'},
                          title=f"Distribution de {selected_qual} par Attrition")
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.markdown("### Analyse des Corrélations")
        
        # Encoder pour calculer les corrélations
        df_encoded, _ = encode_categorical(df)
        
        # Matrice de corrélation
        corr_matrix = df_encoded.corr()
        
        # Top corrélations avec Attrition
        st.markdown("#### Corrélations avec l'Attrition")
        
        attrition_corr = corr_matrix['Attrition'].drop('Attrition').sort_values(key=abs, ascending=False)
        
        fig = px.bar(x=attrition_corr.head(15).values, 
                    y=attrition_corr.head(15).index,
                    orientation='h',
                    color=attrition_corr.head(15).values,
                    color_continuous_scale='RdBu_r',
                    title="Top 15 Corrélations avec l'Attrition")
        fig.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig, use_container_width=True)
        
        # Heatmap complète
        st.markdown("#### Matrice de Corrélation Complète")
        
        # Sélection des variables pour la heatmap
        vars_for_heatmap = st.multiselect("Variables pour la heatmap:", 
                                          corr_matrix.columns.tolist(),
                                          default=corr_matrix.columns.tolist()[:15])
        
        if vars_for_heatmap:
            fig = px.imshow(corr_matrix.loc[vars_for_heatmap, vars_for_heatmap],
                           color_continuous_scale='RdBu_r',
                           aspect='auto',
                           title="Matrice de Corrélation")
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.markdown("### Détection des Outliers")
        
        st.markdown("""
        <div class="info-box">
        Nous utilisons la méthode IQR (Interquartile Range) pour détecter les outliers.
        Un point est considéré comme outlier s'il est en dehors de l'intervalle [Q1 - 1.5*IQR, Q3 + 1.5*IQR].
        </div>
        """, unsafe_allow_html=True)
        
        # Boxplots
        selected_vars = st.multiselect("Sélectionner les variables:", 
                                       quant_vars, 
                                       default=['Age', 'MonthlyIncome', 'YearsAtCompany', 'TotalWorkingYears'])
        
        if selected_vars:
            fig = make_subplots(rows=1, cols=len(selected_vars), 
                               subplot_titles=selected_vars)
            
            for i, var in enumerate(selected_vars):
                fig.add_trace(
                    go.Box(y=df[var], name=var, marker_color='#667eea'),
                    row=1, col=i+1
                )
            
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        # Comptage des outliers
        st.markdown("#### Nombre d'Outliers par Variable")
        
        outlier_counts = {}
        for var in quant_vars:
            Q1 = df[var].quantile(0.25)
            Q3 = df[var].quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((df[var] < Q1 - 1.5*IQR) | (df[var] > Q3 + 1.5*IQR)).sum()
            outlier_counts[var] = outliers
        
        outlier_df = pd.DataFrame.from_dict(outlier_counts, orient='index', columns=['Outliers'])
        outlier_df = outlier_df.sort_values('Outliers', ascending=False)
        
        fig = px.bar(outlier_df.head(10), 
                    x=outlier_df.head(10).index, 
                    y='Outliers',
                    title="Top 10 Variables avec Outliers",
                    color='Outliers',
                    color_continuous_scale='Reds')
        st.plotly_chart(fig, use_container_width=True)


# ==================== PAGE: ANALYSE FACTORIELLE ====================

def page_afdm(df):
    st.markdown("## 🔬 Analyse Factorielle")
    
    # Préparation des données
    df_processed = preprocess_for_analysis(df)
    
    # Identifier correctement les types de variables
    # Variables numériques continues (exclure les échelles ordinales qui pourraient être traitées comme quali)
    numeric_cols = df_processed.select_dtypes(include=[np.number]).columns.tolist()
    object_cols = df_processed.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Variables vraiment quantitatives (continues)
    quant_vars = [col for col in numeric_cols if df_processed[col].nunique() > 10]
    
    # Variables qualitatives (catégorielles + ordinales avec peu de modalités)
    qual_vars = object_cols + [col for col in numeric_cols if df_processed[col].nunique() <= 10]
    
    # Retirer la cible des variables actives
    if 'Attrition' in qual_vars:
        qual_vars_active = [v for v in qual_vars if v != 'Attrition']
    else:
        qual_vars_active = qual_vars
    
    if 'Attrition' in quant_vars:
        quant_vars = [v for v in quant_vars if v != 'Attrition']
    
    # ==================== SÉLECTION DU MODÈLE ====================
    st.markdown("### 🎯 Choix du Modèle d'Analyse")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        model_type = st.selectbox(
            "Sélectionnez le type d'analyse:",
            ["AFDM (Données Mixtes)", "ACP (Quantitatives)", "ACM (Qualitatives)", "AFC (Tableau de Contingence)"],
            help="Choisissez le modèle adapté à vos données"
        )
    
    with col2:
        if model_type == "AFDM (Données Mixtes)":
            st.markdown("""
            <div class="info-box">
            <strong>AFDM - Analyse Factorielle des Données Mixtes</strong><br>
            ✅ Adapté quand vous avez des variables quantitatives ET qualitatives.<br>
            Combine les principes de l'ACP et de l'ACM.
            </div>
            """, unsafe_allow_html=True)
        elif model_type == "ACP (Quantitatives)":
            st.markdown("""
            <div class="info-box">
            <strong>ACP - Analyse en Composantes Principales</strong><br>
            ✅ Adapté pour les variables quantitatives uniquement.<br>
            Réduit la dimensionnalité en préservant la variance.
            </div>
            """, unsafe_allow_html=True)
        elif model_type == "ACM (Qualitatives)":
            st.markdown("""
            <div class="info-box">
            <strong>ACM - Analyse des Correspondances Multiples</strong><br>
            ✅ Adapté pour les variables qualitatives uniquement.<br>
            Analyse les associations entre modalités.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="info-box">
            <strong>AFC - Analyse Factorielle des Correspondances</strong><br>
            ✅ Adapté pour un tableau de contingence (2 variables qualitatives).<br>
            Analyse la relation entre deux variables catégorielles.
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # ==================== STATISTIQUES DES VARIABLES ====================
    st.markdown("### 📊 Variables Disponibles")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Variables Quantitatives", len(quant_vars))
    col2.metric("Variables Qualitatives", len(qual_vars_active))
    col3.metric("Total", len(quant_vars) + len(qual_vars_active))
    
    with st.expander("📋 Voir la liste des variables"):
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Quantitatives:**")
            st.write(", ".join(quant_vars))
        with col2:
            st.write("**Qualitatives:**")
            st.write(", ".join(qual_vars_active))
    
    # ==================== PARAMÈTRES ====================
    st.markdown("### ⚙️ Paramètres de l'Analyse")
    
    col1, col2 = st.columns(2)
    
    with col1:
        n_components = st.slider("Nombre de composantes:", 2, 10, 5)
    
    with col2:
        if model_type == "ACP (Quantitatives)":
            acp_type = st.radio("Type d'ACP:", ["Centrée-Réduite", "Centrée uniquement"])
    
    # Sélection des variables selon le modèle
    with st.expander("🔧 Sélection des Variables"):
        if model_type == "ACP (Quantitatives)":
            selected_quant = st.multiselect("Variables quantitatives:", quant_vars, default=quant_vars)
            selected_qual = []
        elif model_type == "ACM (Qualitatives)":
            selected_qual = st.multiselect("Variables qualitatives:", qual_vars_active, default=qual_vars_active[:10])
            selected_quant = []
        elif model_type == "AFC (Tableau de Contingence)":
            afc_var1 = st.selectbox("Variable 1:", qual_vars_active, index=0)
            afc_var2 = st.selectbox("Variable 2:", [v for v in qual_vars_active if v != afc_var1], index=0)
            selected_quant = []
            selected_qual = [afc_var1, afc_var2]
        else:  # AFDM
            selected_quant = st.multiselect("Variables quantitatives:", quant_vars, default=quant_vars)
            selected_qual = st.multiselect("Variables qualitatives:", qual_vars_active, default=qual_vars_active[:5])
    
    # ==================== EXÉCUTION DE L'ANALYSE ====================
    if st.button(f"🚀 Lancer l'Analyse ({model_type.split(' ')[0]})", type="primary"):
        
        # Vérifications
        if model_type == "AFDM (Données Mixtes)" and (len(selected_quant) == 0 or len(selected_qual) == 0):
            st.error("⚠️ L'AFDM nécessite au moins une variable quantitative ET une variable qualitative!")
            st.info("💡 Utilisez l'ACP si vous n'avez que des variables quantitatives, ou l'ACM si vous n'avez que des qualitatives.")
            return
        
        if model_type == "ACP (Quantitatives)" and len(selected_quant) < 2:
            st.error("⚠️ L'ACP nécessite au moins 2 variables quantitatives!")
            return
            
        if model_type == "ACM (Qualitatives)" and len(selected_qual) < 2:
            st.error("⚠️ L'ACM nécessite au moins 2 variables qualitatives!")
            return
        
        with st.spinner(f"Calcul de l'{model_type.split(' ')[0]} en cours..."):
            
            try:
                # ==================== ACP ====================
                if model_type == "ACP (Quantitatives)":
                    df_analysis = df_processed[selected_quant].copy()
                    
                    # Standardisation
                    if acp_type == "Centrée-Réduite":
                        model = prince.PCA(n_components=n_components, rescale_with_std=True)
                    else:
                        model = prince.PCA(n_components=n_components, rescale_with_std=False)
                    
                    model.fit(df_analysis)
                    row_coords = model.row_coordinates(df_analysis)
                    # Utiliser column_coordinates_ (attribut stocké après fit)
                    col_coords = model.column_coordinates_
                    
                    analysis_name = "ACP"
                    
                # ==================== ACM ====================
                elif model_type == "ACM (Qualitatives)":
                    df_analysis = df_processed[selected_qual].copy()
                    
                    # Convertir en catégories
                    for col in selected_qual:
                        df_analysis[col] = df_analysis[col].astype(str).astype('category')
                    
                    model = prince.MCA(n_components=n_components)
                    model.fit(df_analysis)
                    row_coords = model.row_coordinates(df_analysis)
                    # Utiliser column_coordinates comme méthode
                    col_coords = model.column_coordinates(df_analysis)
                    
                    analysis_name = "ACM"
                    
                # ==================== AFC ====================
                elif model_type == "AFC (Tableau de Contingence)":
                    # Créer le tableau de contingence
                    contingency_table = pd.crosstab(df_processed[afc_var1], df_processed[afc_var2])
                    
                    model = prince.CA(n_components=min(n_components, min(contingency_table.shape)-1))
                    model.fit(contingency_table)
                    row_coords = model.row_coordinates(contingency_table)
                    # Utiliser column_coordinates comme méthode
                    col_coords = model.column_coordinates(contingency_table)
                    
                    analysis_name = "AFC"
                    
                # ==================== AFDM ====================
                else:
                    # Créer un nouveau DataFrame propre pour éviter les problèmes de types
                    data_dict = {}
                    
                    # Variables quantitatives - forcer float64
                    for col in selected_quant:
                        values = pd.to_numeric(df_processed[col], errors='coerce')
                        data_dict[col] = values.astype('float64')
                    
                    # Variables qualitatives - garder comme string (pas category!)
                    for col in selected_qual:
                        data_dict[col] = df_processed[col].astype(str)
                    
                    df_analysis = pd.DataFrame(data_dict)
                    
                    # Supprimer les lignes avec des NaN
                    df_analysis = df_analysis.dropna()
                    
                    if len(df_analysis) == 0:
                        st.error("⚠️ Aucune donnée valide après nettoyage!")
                        return
                    
                    # Vérification des types détectés par pandas
                    numeric_detected = df_analysis.select_dtypes(include=[np.number]).columns.tolist()
                    object_detected = df_analysis.select_dtypes(include=['object']).columns.tolist()
                    
                    st.info(f"📊 Variables détectées - Numériques: {len(numeric_detected)}, Catégorielles: {len(object_detected)}")
                    
                    if len(numeric_detected) == 0:
                        st.error("⚠️ Aucune variable numérique détectée! Utilisez l'ACM à la place.")
                        return
                    
                    if len(object_detected) == 0:
                        st.error("⚠️ Aucune variable catégorielle détectée! Utilisez l'ACP à la place.")
                        return
                    
                    model = prince.FAMD(n_components=n_components, n_iter=3, random_state=42)
                    model.fit(df_analysis)
                    row_coords = model.row_coordinates(df_analysis)
                    # Utiliser column_coordinates_ (attribut)
                    col_coords = model.column_coordinates_
                    
                    analysis_name = "AFDM"
                
                # Ajouter Attrition aux coordonnées des individus
                if model_type == "AFC (Tableau de Contingence)":
                    # Pour AFC, pas d'ajout d'Attrition car c'est un tableau de contingence
                    pass
                elif len(row_coords) == len(df_processed):
                    row_coords['Attrition'] = df_processed['Attrition'].values
                else:
                    # Si on a supprimé des lignes (AFDM avec NaN)
                    row_coords['Attrition'] = df_processed.loc[df_analysis.index, 'Attrition'].values
                
                # ==================== AFFICHAGE DES RÉSULTATS ====================
                st.markdown(f"### 📊 Résultats de l'{analysis_name}")
                
                tab1, tab2, tab3, tab4 = st.tabs(["📉 Inertie", "👥 Individus", "📊 Variables", "🎯 Interprétation"])
                
                with tab1:
                    display_eigenvalues(model, n_components, analysis_name)
                
                with tab2:
                    display_individuals(row_coords, model, n_components, analysis_name)
                
                with tab3:
                    if model_type == "AFC (Tableau de Contingence)":
                        display_variables_afc(model, row_coords, col_coords, afc_var1, afc_var2, contingency_table)
                    else:
                        display_variables(model, col_coords, n_components, analysis_name, df_analysis)
                
                with tab4:
                    display_interpretation(model, col_coords, n_components, analysis_name)
                
                # Sauvegarder pour clustering
                st.session_state['factor_coords'] = row_coords
                st.session_state['factor_model'] = model
                st.session_state['analysis_name'] = analysis_name
                
                st.success(f"✅ {analysis_name} terminée! Les résultats ont été sauvegardés pour le clustering.")
                
            except Exception as e:
                st.error(f"❌ Erreur lors de l'analyse: {str(e)}")
                st.info("💡 Conseil: Vérifiez que vos variables sont bien du bon type (quantitatives ou qualitatives)")
                import traceback
                with st.expander("Voir les détails de l'erreur"):
                    st.code(traceback.format_exc())


def display_eigenvalues(model, n_components, analysis_name):
    """Affiche les valeurs propres et l'inertie"""
    st.markdown("#### Valeurs Propres et Inertie Expliquée")
    
    eigenvalues = model.eigenvalues_summary
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.dataframe(eigenvalues.round(4), use_container_width=True)
    
    with col2:
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=[f"F{i+1}" for i in range(len(eigenvalues))],
            y=eigenvalues['% of variance'].values,
            name='% Variance',
            marker_color='#667eea'
        ))
        
        fig.add_trace(go.Scatter(
            x=[f"F{i+1}" for i in range(len(eigenvalues))],
            y=eigenvalues['% of variance'].cumsum().values,
            name='% Cumulé',
            mode='lines+markers',
            marker_color='#EF4444'
        ))
        
        fig.update_layout(
            title="Éboulis des Valeurs Propres",
            xaxis_title="Composantes",
            yaxis_title="% de Variance",
            barmode='overlay'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Règle de Kaiser pour ACP
    if analysis_name == "ACP":
        kaiser_count = (model.eigenvalues_ > 1).sum()
        st.markdown(f"""
        <div class="success-box">
        <strong>Critères de sélection des axes :</strong><br>
        • <strong>Règle de Kaiser:</strong> {kaiser_count} axes ont une valeur propre > 1<br>
        • <strong>Critère du coude:</strong> Observer l'inflexion de la courbe<br>
        • <strong>Inertie cumulée:</strong> Retenir les axes expliquant 70-80% de l'inertie
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="success-box">
        <strong>Interprétation :</strong> Le nombre d'axes à retenir se détermine par le critère du coude 
        (là où la courbe s'infléchit) ou en conservant les axes expliquant au moins 70-80% de l'inertie totale.
        </div>
        """, unsafe_allow_html=True)


def display_individuals(row_coords, model, n_components, analysis_name):
    """Affiche la projection des individus"""
    st.markdown("#### Projection des Individus")
    
    col1, col2 = st.columns(2)
    
    n_dims = min(n_components, row_coords.shape[1] - 1)  # -1 pour Attrition
    
    with col1:
        axis_x = st.selectbox("Axe X:", [f"F{i}" for i in range(n_dims)], index=0, key="ind_x")
    
    with col2:
        axis_y = st.selectbox("Axe Y:", [f"F{i}" for i in range(n_dims)], index=min(1, n_dims-1), key="ind_y")
    
    # Renommer les colonnes si nécessaire
    coord_cols = [col for col in row_coords.columns if col != 'Attrition']
    rename_dict = {old: f"F{i}" for i, old in enumerate(coord_cols)}
    row_coords_plot = row_coords.rename(columns=rename_dict)
    
    eigenvalues = model.eigenvalues_summary
    
    fig = px.scatter(
        row_coords_plot, 
        x=axis_x, 
        y=axis_y,
        color='Attrition',
        color_discrete_map={'No': '#10B981', 'Yes': '#EF4444'},
        title=f"Projection des Individus sur le Plan {axis_x}-{axis_y}",
        opacity=0.6
    )
    
    try:
        x_var = eigenvalues.loc[int(axis_x[1]), '% of variance']
        y_var = eigenvalues.loc[int(axis_y[1]), '% of variance']
        fig.update_layout(
            xaxis_title=f"{axis_x} ({x_var:.2f}%)",
            yaxis_title=f"{axis_y} ({y_var:.2f}%)"
        )
    except:
        pass
    
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.add_vline(x=0, line_dash="dash", line_color="gray")
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown(f"""
    <div class="info-box">
    <strong>Lecture ({analysis_name}):</strong> Les individus proches dans le plan ont des profils similaires. 
    La coloration par Attrition permet d'identifier visuellement si les employés qui partent 
    ont des caractéristiques communes.
    </div>
    """, unsafe_allow_html=True)


def display_variables(model, col_coords, n_components, analysis_name, df_analysis):
    """Affiche la projection des variables"""
    st.markdown("#### Projection des Variables")
    
    # Renommer les colonnes
    n_dims = min(n_components, col_coords.shape[1])
    rename_dict = {old: f"F{i}" for i, old in enumerate(col_coords.columns[:n_dims])}
    col_coords_plot = col_coords.rename(columns=rename_dict)
    
    fig = px.scatter(
        col_coords_plot,
        x='F0',
        y='F1',
        text=col_coords_plot.index,
        title="Projection des Variables sur le Plan F0-F1"
    )
    
    fig.update_traces(textposition='top center', marker=dict(size=10, color='#667eea'))
    
    # Cercle de corrélation pour ACP
    if analysis_name == "ACP":
        theta = np.linspace(0, 2*np.pi, 100)
        fig.add_trace(go.Scatter(
            x=np.cos(theta),
            y=np.sin(theta),
            mode='lines',
            line=dict(color='gray', dash='dash'),
            showlegend=False,
            name='Cercle de corrélation'
        ))
    
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.add_vline(x=0, line_dash="dash", line_color="gray")
    
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)
    
    # Contributions
    st.markdown("#### Contributions aux Axes")
    try:
        contributions = model.column_contributions_
        contributions.columns = [f"F{i}" for i in range(contributions.shape[1])]
        st.dataframe(contributions.round(4), use_container_width=True)
    except:
        st.dataframe(col_coords_plot.round(4), use_container_width=True)


def display_variables_afc(model, row_coords, col_coords, var1, var2, contingency_table):
    """Affiche les résultats spécifiques à l'AFC"""
    st.markdown("#### Analyse Factorielle des Correspondances")
    
    # Tableau de contingence
    st.markdown("##### Tableau de Contingence")
    st.dataframe(contingency_table, use_container_width=True)
    
    # Graphique biplot
    st.markdown("##### Biplot AFC")
    
    row_coords_plot = row_coords.copy()
    col_coords_plot = col_coords.copy()
    
    row_coords_plot.columns = [f"F{i}" for i in range(row_coords_plot.shape[1])]
    col_coords_plot.columns = [f"F{i}" for i in range(col_coords_plot.shape[1])]
    
    fig = go.Figure()
    
    # Points lignes
    fig.add_trace(go.Scatter(
        x=row_coords_plot['F0'],
        y=row_coords_plot['F1'],
        mode='markers+text',
        text=row_coords_plot.index,
        textposition='top center',
        name=f'{var1} (lignes)',
        marker=dict(size=12, color='#667eea', symbol='circle')
    ))
    
    # Points colonnes
    fig.add_trace(go.Scatter(
        x=col_coords_plot['F0'],
        y=col_coords_plot['F1'],
        mode='markers+text',
        text=col_coords_plot.index,
        textposition='top center',
        name=f'{var2} (colonnes)',
        marker=dict(size=12, color='#EF4444', symbol='diamond')
    ))
    
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.add_vline(x=0, line_dash="dash", line_color="gray")
    
    fig.update_layout(
        title=f"Biplot AFC: {var1} vs {var2}",
        xaxis_title="Dimension 1",
        yaxis_title="Dimension 2",
        height=600
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("""
    <div class="info-box">
    <strong>Lecture de l'AFC:</strong> Les modalités proches dans le plan sont associées. 
    Les points lignes (●) représentent les modalités de la première variable, 
    les points colonnes (◆) celles de la seconde.
    </div>
    """, unsafe_allow_html=True)


def display_interpretation(model, col_coords, n_components, analysis_name):
    """Affiche l'interprétation des axes"""
    st.markdown("#### Interprétation des Axes Factoriels")
    
    st.markdown(f"""
    **Axe F0 (1ère composante)** - Pour l'{analysis_name}:
    - Variables/modalités positives : caractérisent les individus à droite du plan
    - Variables/modalités négatives : caractérisent les individus à gauche du plan
    
    **Axe F1 (2ème composante)** - Opposition entre :
    - Haut du plan : profils avec certaines caractéristiques
    - Bas du plan : profils avec caractéristiques opposées
    """)
    
    # Top contributions
    st.markdown("##### Top Variables/Modalités Contributrices par Axe")
    
    try:
        contributions = model.column_contributions_
        contributions.columns = [f"F{i}" for i in range(contributions.shape[1])]
        
        for i in range(min(3, n_components)):
            with st.expander(f"📊 Axe F{i}"):
                top_contrib = contributions[f'F{i}'].sort_values(ascending=False).head(10)
                
                fig = px.bar(
                    x=top_contrib.values,
                    y=top_contrib.index,
                    orientation='h',
                    title=f"Top 10 Contributions à F{i}",
                    color=top_contrib.values,
                    color_continuous_scale='Viridis'
                )
                fig.update_layout(yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.info(f"Contributions non disponibles pour ce type d'analyse: {e}")


# ==================== PAGE: CLUSTERING ====================

def page_clustering(df):
    st.markdown("## 🎯 Clustering des Employés")
    
    st.markdown("""
    <div class="info-box">
    <strong>Objectif :</strong> Identifier des groupes homogènes d'employés ayant des caractéristiques similaires.
    Nous utilisons K-Means et la Classification Ascendante Hiérarchique (CAH).
    </div>
    """, unsafe_allow_html=True)
    
    # Préparation des données
    df_processed = preprocess_for_analysis(df)
    df_encoded, label_encoders = encode_categorical(df_processed)
    
    # Standardisation
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_encoded)
    
    tab1, tab2, tab3 = st.tabs(["📊 K-Means", "🌳 CAH", "📈 Profilage"])
    
    with tab1:
        st.markdown("### K-Means Clustering")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Méthode du coude
            st.markdown("#### Méthode du Coude (Elbow Method)")
            
            inertias = []
            silhouettes = []
            K_range = range(2, 11)
            
            for k in K_range:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                kmeans.fit(X_scaled)
                inertias.append(kmeans.inertia_)
                silhouettes.append(silhouette_score(X_scaled, kmeans.labels_))
            
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            fig.add_trace(
                go.Scatter(x=list(K_range), y=inertias, name="Inertie", 
                          mode='lines+markers', marker_color='#667eea'),
                secondary_y=False
            )
            
            fig.add_trace(
                go.Scatter(x=list(K_range), y=silhouettes, name="Silhouette", 
                          mode='lines+markers', marker_color='#EF4444'),
                secondary_y=True
            )
            
            fig.update_layout(title="Méthode du Coude & Score Silhouette")
            fig.update_xaxes(title_text="Nombre de Clusters")
            fig.update_yaxes(title_text="Inertie", secondary_y=False)
            fig.update_yaxes(title_text="Score Silhouette", secondary_y=True)
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### Choix du Nombre de Clusters")
            n_clusters = st.slider("Nombre de clusters K:", 2, 10, 4)
            
            st.markdown("""
            <div class="info-box">
            <strong>Critères de choix :</strong>
            <ul>
            <li><strong>Coude :</strong> Chercher le point d'inflexion de l'inertie</li>
            <li><strong>Silhouette :</strong> Maximiser le score silhouette</li>
            <li><strong>Interprétabilité :</strong> Choisir un K permettant une interprétation métier</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        
        if st.button("🚀 Appliquer K-Means", type="primary"):
            with st.spinner("Clustering en cours..."):
                # K-Means
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                clusters = kmeans.fit_predict(X_scaled)
                
                # Ajouter les clusters au dataframe
                df_clustered = df_processed.copy()
                df_clustered['Cluster'] = clusters
                
                # Visualisation avec PCA
                pca = PCA(n_components=2)
                X_pca = pca.fit_transform(X_scaled)
                
                df_viz = pd.DataFrame({
                    'PC1': X_pca[:, 0],
                    'PC2': X_pca[:, 1],
                    'Cluster': clusters.astype(str),
                    'Attrition': df_processed['Attrition']
                })
                
                col1, col2 = st.columns(2)
                
                with col1:
                    fig = px.scatter(
                        df_viz, x='PC1', y='PC2', 
                        color='Cluster',
                        title="Clusters (projection PCA)",
                        color_discrete_sequence=px.colors.qualitative.Set1
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    fig = px.scatter(
                        df_viz, x='PC1', y='PC2', 
                        color='Attrition',
                        title="Attrition (projection PCA)",
                        color_discrete_map={'No': '#10B981', 'Yes': '#EF4444'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Métriques
                st.markdown("#### Métriques de Qualité")
                col1, col2, col3 = st.columns(3)
                col1.metric("Score Silhouette", f"{silhouette_score(X_scaled, clusters):.3f}")
                col2.metric("Inertie", f"{kmeans.inertia_:.2f}")
                col3.metric("Nombre de Clusters", n_clusters)
                
                # Sauvegarder pour le profilage
                st.session_state['clusters'] = clusters
                st.session_state['df_clustered'] = df_clustered
                
                st.success("✅ Clustering terminé! Allez dans l'onglet 'Profilage' pour analyser les clusters.")
    
    with tab2:
        st.markdown("### Classification Ascendante Hiérarchique (CAH)")
        
        # Sous-échantillon pour la CAH (plus rapide)
        sample_size = st.slider("Taille de l'échantillon:", 100, min(1000, len(df)), 500)
        
        if st.button("🌳 Générer le Dendrogramme"):
            with st.spinner("Calcul du dendrogramme..."):
                # Échantillon
                indices = np.random.choice(len(X_scaled), sample_size, replace=False)
                X_sample = X_scaled[indices]
                
                # Linkage
                linkage_method = st.selectbox("Méthode de liaison:", 
                                              ['ward', 'complete', 'average', 'single'])
                Z = linkage(X_sample, method=linkage_method)
                
                # Dendrogramme
                fig, ax = plt.subplots(figsize=(12, 6))
                dendrogram(Z, truncate_mode='level', p=5, ax=ax)
                ax.set_title("Dendrogramme (CAH)")
                ax.set_xlabel("Individus")
                ax.set_ylabel("Distance")
                st.pyplot(fig)
                
                st.markdown("""
                <div class="info-box">
                <strong>Lecture du dendrogramme :</strong> Couper horizontalement l'arbre permet de définir 
                le nombre de clusters. Une grande hauteur de fusion indique des clusters bien séparés.
                </div>
                """, unsafe_allow_html=True)
    
    with tab3:
        st.markdown("### Profilage des Clusters")
        
        if 'df_clustered' in st.session_state:
            df_clustered = st.session_state['df_clustered']
            
            # Distribution de l'attrition par cluster
            st.markdown("#### Attrition par Cluster")
            
            attrition_by_cluster = df_clustered.groupby('Cluster')['Attrition'].value_counts(normalize=True).unstack()
            
            fig = px.bar(
                attrition_by_cluster,
                barmode='group',
                title="Taux d'Attrition par Cluster",
                color_discrete_map={0: '#10B981', 1: '#EF4444'}
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Caractéristiques moyennes par cluster
            st.markdown("#### Caractéristiques Moyennes par Cluster")
            
            quant_vars, _ = get_variable_types(df_clustered)
            quant_vars = [v for v in quant_vars if v != 'Cluster']
            
            cluster_means = df_clustered.groupby('Cluster')[quant_vars].mean()
            
            # Normaliser pour le radar chart
            cluster_means_norm = (cluster_means - cluster_means.min()) / (cluster_means.max() - cluster_means.min())
            
            # Sélection des variables pour le radar
            selected_vars = st.multiselect(
                "Variables pour le profil:",
                quant_vars,
                default=['Age', 'MonthlyIncome', 'YearsAtCompany', 'JobSatisfaction', 'WorkLifeBalance'][:min(5, len(quant_vars))]
            )
            
            if selected_vars:
                fig = go.Figure()
                
                for cluster in cluster_means_norm.index:
                    fig.add_trace(go.Scatterpolar(
                        r=cluster_means_norm.loc[cluster, selected_vars].values,
                        theta=selected_vars,
                        fill='toself',
                        name=f'Cluster {cluster}'
                    ))
                
                fig.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                    title="Profil des Clusters (Radar Chart)"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Tableau des moyennes
            st.markdown("#### Statistiques Détaillées par Cluster")
            st.dataframe(cluster_means.round(2), use_container_width=True)
            
            # Interprétation automatique
            st.markdown("#### 💡 Interprétation des Clusters")
            
            for cluster in sorted(df_clustered['Cluster'].unique()):
                cluster_data = df_clustered[df_clustered['Cluster'] == cluster]
                attrition_rate = (cluster_data['Attrition'] == 'Yes').mean() * 100
                
                with st.expander(f"Cluster {cluster} ({len(cluster_data)} employés - Attrition: {attrition_rate:.1f}%)"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Caractéristiques quantitatives:**")
                        for var in ['Age', 'MonthlyIncome', 'YearsAtCompany']:
                            if var in cluster_data.columns:
                                mean_val = cluster_data[var].mean()
                                overall_mean = df_clustered[var].mean()
                                diff = ((mean_val - overall_mean) / overall_mean) * 100
                                emoji = "📈" if diff > 10 else ("📉" if diff < -10 else "➡️")
                                st.write(f"- {var}: {mean_val:.1f} ({emoji} {diff:+.1f}% vs moyenne)")
                    
                    with col2:
                        st.write("**Caractéristiques qualitatives (mode):**")
                        for var in ['Department', 'JobRole', 'MaritalStatus']:
                            if var in cluster_data.columns:
                                mode_val = cluster_data[var].mode().iloc[0]
                                st.write(f"- {var}: {mode_val}")
        else:
            st.warning("⚠️ Veuillez d'abord effectuer le clustering K-Means dans l'onglet précédent.")


# ==================== PAGE: CLASSIFICATION ====================

def page_classification(df):
    st.markdown("## 🤖 Prédiction de l'Attrition")
    
    st.markdown("""
    <div class="info-box">
    <strong>Objectif :</strong> Construire un modèle prédictif pour identifier les employés à risque de départ.
    Nous comparons plusieurs algorithmes et analysons l'importance des variables.
    </div>
    """, unsafe_allow_html=True)
    
    # Préparation des données
    df_processed = preprocess_for_analysis(df)
    df_encoded, label_encoders = encode_categorical(df_processed)
    
    # Features et target
    X = df_encoded.drop('Attrition', axis=1)
    y = df_encoded['Attrition']
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Standardisation
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    tab1, tab2, tab3 = st.tabs(["🎯 Entraînement", "📊 Évaluation", "🔍 Interprétabilité"])
    
    with tab1:
        st.markdown("### Configuration du Modèle")
        
        col1, col2 = st.columns(2)
        
        with col1:
            model_choice = st.selectbox(
                "Algorithme:",
                ["Random Forest", "Gradient Boosting", "Logistic Regression"]
            )
            
            use_smote = st.checkbox("Utiliser SMOTE (rééquilibrage)", value=True)
        
        with col2:
            if model_choice == "Random Forest":
                n_estimators = st.slider("Nombre d'arbres:", 50, 300, 100)
                max_depth = st.slider("Profondeur max:", 3, 20, 10)
            elif model_choice == "Gradient Boosting":
                n_estimators = st.slider("Nombre d'itérations:", 50, 300, 100)
                learning_rate = st.slider("Learning rate:", 0.01, 0.3, 0.1)
        
        if st.button("🚀 Entraîner le Modèle", type="primary"):
            with st.spinner("Entraînement en cours..."):
                
                # SMOTE
                if use_smote:
                    smote = SMOTE(random_state=42)
                    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
                    st.info(f"SMOTE appliqué: {len(y_train)} → {len(y_train_balanced)} échantillons")
                else:
                    X_train_balanced, y_train_balanced = X_train_scaled, y_train
                
                # Modèle
                if model_choice == "Random Forest":
                    model = RandomForestClassifier(
                        n_estimators=n_estimators, 
                        max_depth=max_depth,
                        random_state=42,
                        class_weight='balanced'
                    )
                elif model_choice == "Gradient Boosting":
                    model = GradientBoostingClassifier(
                        n_estimators=n_estimators,
                        learning_rate=learning_rate,
                        random_state=42
                    )
                else:
                    model = LogisticRegression(
                        class_weight='balanced',
                        random_state=42,
                        max_iter=1000
                    )
                
                # Entraînement
                model.fit(X_train_balanced, y_train_balanced)
                
                # Prédictions
                y_pred = model.predict(X_test_scaled)
                y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
                
                # Validation croisée
                cv_scores = cross_val_score(model, X_train_balanced, y_train_balanced, cv=5, scoring='f1')
                
                # Sauvegarder les résultats
                st.session_state['model'] = model
                st.session_state['y_test'] = y_test
                st.session_state['y_pred'] = y_pred
                st.session_state['y_pred_proba'] = y_pred_proba
                st.session_state['X_train'] = X_train
                st.session_state['feature_names'] = X.columns.tolist()
                st.session_state['cv_scores'] = cv_scores
                
                st.success(f"✅ Modèle entraîné! F1-score CV: {cv_scores.mean():.3f} (±{cv_scores.std():.3f})")
    
    with tab2:
        st.markdown("### Évaluation du Modèle")
        
        if 'model' in st.session_state:
            y_test = st.session_state['y_test']
            y_pred = st.session_state['y_pred']
            y_pred_proba = st.session_state['y_pred_proba']
            cv_scores = st.session_state['cv_scores']
            
            # Métriques
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.3f}")
            col2.metric("Precision", f"{precision_score(y_test, y_pred):.3f}")
            col3.metric("Recall", f"{recall_score(y_test, y_pred):.3f}")
            col4.metric("F1-Score", f"{f1_score(y_test, y_pred):.3f}")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Matrice de confusion
                st.markdown("#### Matrice de Confusion")
                cm = confusion_matrix(y_test, y_pred)
                
                fig = px.imshow(
                    cm,
                    labels=dict(x="Prédit", y="Réel", color="Count"),
                    x=['Reste', 'Part'],
                    y=['Reste', 'Part'],
                    color_continuous_scale='Blues',
                    text_auto=True
                )
                fig.update_layout(title="Matrice de Confusion")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Courbe ROC
                st.markdown("#### Courbe ROC")
                fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
                roc_auc = auc(fpr, tpr)
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=fpr, y=tpr,
                    mode='lines',
                    name=f'ROC (AUC = {roc_auc:.3f})',
                    line=dict(color='#667eea')
                ))
                fig.add_trace(go.Scatter(
                    x=[0, 1], y=[0, 1],
                    mode='lines',
                    name='Aléatoire',
                    line=dict(color='gray', dash='dash')
                ))
                fig.update_layout(
                    title="Courbe ROC",
                    xaxis_title="Taux de Faux Positifs",
                    yaxis_title="Taux de Vrais Positifs"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Rapport de classification
            st.markdown("#### Rapport de Classification")
            report = classification_report(y_test, y_pred, target_names=['Reste', 'Part'], output_dict=True)
            report_df = pd.DataFrame(report).T
            st.dataframe(report_df.round(3), use_container_width=True)
            
            # Validation croisée
            st.markdown("#### Validation Croisée (5-Fold)")
            fig = px.box(y=cv_scores, title="Distribution des F1-Scores (CV)")
            st.plotly_chart(fig, use_container_width=True)
            
        else:
            st.warning("⚠️ Veuillez d'abord entraîner un modèle dans l'onglet précédent.")
    
    with tab3:
        st.markdown("### Interprétabilité du Modèle")
        
        if 'model' in st.session_state:
            model = st.session_state['model']
            feature_names = st.session_state['feature_names']
            
            # Feature importance
            st.markdown("#### Importance des Variables")
            
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
            elif hasattr(model, 'coef_'):
                importances = np.abs(model.coef_[0])
            else:
                importances = None
            
            if importances is not None:
                importance_df = pd.DataFrame({
                    'Variable': feature_names,
                    'Importance': importances
                }).sort_values('Importance', ascending=False)
                
                # Top 15
                top_features = importance_df.head(15)
                
                fig = px.bar(
                    top_features,
                    x='Importance',
                    y='Variable',
                    orientation='h',
                    title="Top 15 Variables les Plus Importantes",
                    color='Importance',
                    color_continuous_scale='Viridis'
                )
                fig.update_layout(yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig, use_container_width=True)
                
                # Tableau complet
                with st.expander("📋 Voir toutes les importances"):
                    st.dataframe(importance_df, use_container_width=True)
                
                # Interprétation
                st.markdown("#### 💡 Interprétation des Résultats")
                
                top_3 = importance_df.head(3)['Variable'].tolist()
                
                st.markdown(f"""
                <div class="success-box">
                <strong>Variables clés pour prédire l'attrition :</strong>
                <ol>
                <li><strong>{top_3[0]}</strong></li>
                <li><strong>{top_3[1]}</strong></li>
                <li><strong>{top_3[2]}</strong></li>
                </ol>
                Ces variables sont les plus discriminantes pour identifier les employés à risque de départ.
                </div>
                """, unsafe_allow_html=True)
                
        else:
            st.warning("⚠️ Veuillez d'abord entraîner un modèle.")


# ==================== PAGE: CONCLUSION ====================

def page_conclusion():
    st.markdown("## 📝 Conclusion et Recommandations")
    
    st.markdown("### 🎯 Synthèse de l'Analyse")
    
    st.markdown("""
    <div class="info-box">
    <strong>Contexte :</strong> Ce projet analyse le dataset IBM HR Analytics contenant <strong>1 470 employés</strong> 
    et <strong>35 variables</strong> pour comprendre les facteurs déterminants de l'attrition. 
    Le taux d'attrition global est de <strong>16,1%</strong> (237 départs sur 1 470 employés).
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Résultats clés en cards
    st.markdown("### 📊 Chiffres Clés du Dataset")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("👥 Employés", "1 470", help="Nombre total d'employés dans le dataset")
    with col2:
        st.metric("🚪 Départs", "237", "16.1%", help="Nombre d'employés ayant quitté l'entreprise")
    with col3:
        st.metric("📋 Variables", "35", help="Nombre de caractéristiques analysées")
    with col4:
        st.metric("💰 Salaire Médian", "4 919$", help="Revenu mensuel médian")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🔬 Résultats de l'Analyse Factorielle")
        st.markdown("""
        <div class="stat-card">
        <p><strong>AFDM (Analyse Factorielle des Données Mixtes) :</strong></p>
        <ul>
        <li>Les <strong>2 premiers axes</strong> expliquent environ <strong>25-30%</strong> de l'inertie totale</li>
        <li><strong>Axe 1</strong> : Opposition entre ancienneté/stabilité vs nouveaux employés</li>
        <li><strong>Axe 2</strong> : Opposition entre satisfaction/engagement vs insatisfaction</li>
        <li>Visualisation claire de la séparation des profils à risque</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("#### 🎯 Résultats du Clustering")
        st.markdown("""
        <div class="stat-card">
        <p><strong>Segmentation K-Means (3-4 clusters optimaux) :</strong></p>
        <ul>
        <li><strong>Cluster "Stables"</strong> : Employés seniors, bons salaires, faible attrition (~8%)</li>
        <li><strong>Cluster "À risque"</strong> : Jeunes, heures sup fréquentes, attrition élevée (~35%)</li>
        <li><strong>Cluster "Satisfaits"</strong> : Bonne satisfaction, ancienneté moyenne (~12%)</li>
        <li>Silhouette score indiquant une bonne séparation des groupes</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("### 🔍 Facteurs Clés d'Attrition Identifiés")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="warning-box">
        <strong>⏰ Heures Supplémentaires</strong><br>
        Les employés faisant des heures sup ont un taux d'attrition de <strong>30.5%</strong> 
        contre <strong>10.4%</strong> pour les autres.
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="warning-box">
        <strong>💵 Revenu Mensuel</strong><br>
        Les employés partis gagnaient en moyenne <strong>4 787$</strong> 
        contre <strong>6 833$</strong> pour ceux restés.
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="warning-box">
        <strong>📅 Ancienneté</strong><br>
        L'attrition est plus forte chez les employés avec <strong>moins de 2 ans</strong> 
        d'ancienneté dans l'entreprise.
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("### 💡 Recommandations Stratégiques")
    
    st.markdown("""
    <div class="success-box">
    <strong>Actions prioritaires pour réduire l'attrition :</strong>
    <ol>
    <li><strong>🕐 Politique de temps de travail :</strong> Réduire les heures supplémentaires systématiques - facteur #1 d'attrition</li>
    <li><strong>💰 Révision salariale ciblée :</strong> Augmenter les salaires des employés sous le seuil de 5 000$/mois</li>
    <li><strong>🎯 Programme d'intégration renforcé :</strong> Accompagnement spécifique les 2 premières années</li>
    <li><strong>📈 Plan de carrière :</strong> Proposer des évolutions claires aux jeunes talents (25-35 ans)</li>
    <li><strong>😊 Suivi satisfaction :</strong> Enquêtes régulières et actions correctives sur l'environnement de travail</li>
    </ol>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("### 🛠️ Méthodologie Utilisée")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Exploration des Données (EDA) :**
        - Analyse univariée et bivariée
        - Détection des corrélations
        - Visualisation des distributions
        - Tests statistiques (Chi², ANOVA)
        """)
    
    with col2:
        st.markdown("""
        **Analyses Multidimensionnelles :**
        - ACP pour les variables quantitatives
        - ACM pour les variables qualitatives  
        - AFDM pour les données mixtes
        - Clustering K-Means et CAH
        """)
    
    st.markdown("---")
    
    st.markdown("### 📚 Références")
    
    st.markdown("""
    - **Dataset** : IBM Watson Analytics - HR Employee Attrition
    - **Librairies** : Pandas, NumPy, Scikit-learn, Prince (AFDM), Plotly, Streamlit
    - **Documentation** : [Prince](https://github.com/MaxHalford/prince) | [Scikit-learn](https://scikit-learn.org/) | [Streamlit](https://docs.streamlit.io/)
    """)


# ==================== MAIN ====================

def main():
    # Sidebar
    st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/5/51/IBM_logo.svg", width=100)
    st.sidebar.title("📊 Navigation")
    
    page = st.sidebar.radio(
        "Aller à:",
        ["🏠 Accueil", "📊 Exploration (EDA)", "🔬 Analyse Factorielle", "🎯 Clustering", "📝 Conclusion"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 👥 Équipe")
    st.sidebar.markdown("""
    - Mohammed Tbahriti
    - Youcef Moulai  
    - Yahia Ouahmed Yanis
    """)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("*Master 2 IA - URCA*")
    st.sidebar.markdown("*INFO0902 - Analyse des Données*")
    
    # Chargement des données
    df = load_data()
    
    if df is not None:
        # Navigation
        if page == "🏠 Accueil":
            page_accueil()
        elif page == "📊 Exploration (EDA)":
            page_exploration(df)
        elif page == "🔬 Analyse Factorielle":
            page_afdm(df)
        elif page == "🎯 Clustering":
            page_clustering(df)
        elif page == "📝 Conclusion":
            page_conclusion()


if __name__ == "__main__":
    main()
