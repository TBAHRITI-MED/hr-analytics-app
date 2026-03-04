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

import prince
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy import stats
from imblearn.over_sampling import SMOTE

import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="HR Analytics - Attrition Analysis",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stApp { background-color: #FAFBFC; }
    .main-header { font-size: 2.5rem; font-weight: 700; color: #1a1a2e; text-align: center; margin-bottom: 0.5rem; }
    .sub-header { font-size: 1.1rem; color: #666; text-align: center; margin-bottom: 2rem; }
    section[data-testid="stSidebar"] { background-color: #f8f9fa !important; border-right: 1px solid #e9ecef; }
    section[data-testid="stSidebar"] > div { background-color: #f8f9fa !important; }
    .info-box { background-color: #e3f2fd; padding: 1rem 1.25rem; border-radius: 8px; border-left: 4px solid #2196F3; margin: 1rem 0; color: #1565c0; }
    .warning-box { background-color: #fff3e0; padding: 1rem 1.25rem; border-radius: 8px; border-left: 4px solid #ff9800; margin: 1rem 0; color: #e65100; }
    .success-box { background-color: #e8f5e9; padding: 1rem 1.25rem; border-radius: 8px; border-left: 4px solid #4caf50; margin: 1rem 0; color: #2e7d32; }
    .stat-card { background: white; padding: 1.25rem; border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); border: 1px solid #eee; margin: 0.5rem 0; }
    .metric-card { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1.5rem; border-radius: 12px; color: white; text-align: center; }
    .stTabs [data-baseweb="tab-list"] { gap: 4px; background-color: #f1f3f4; padding: 4px; border-radius: 8px; }
    .stTabs [data-baseweb="tab"] { background-color: transparent; border-radius: 6px; padding: 8px 16px; color: #5f6368; }
    .stTabs [aria-selected="true"] { background-color: white !important; color: #1a73e8 !important; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
    .stButton > button { background-color: #1a73e8; color: white; border: none; border-radius: 6px; padding: 0.5rem 1.5rem; font-weight: 500; }
    .stButton > button:hover { background-color: #1557b0; }
    [data-testid="stMetricValue"] { font-size: 1.8rem; font-weight: 600; color: #1a1a2e; }
    [data-testid="stMetricLabel"] { color: #666; }
    h2 { color: #1a1a2e; font-weight: 600; border-bottom: 2px solid #1a73e8; padding-bottom: 0.5rem; display: inline-block; }
    h3 { color: #333; font-weight: 600; }
    h4 { color: #444; font-weight: 500; }
    .stDataFrame { border-radius: 8px; border: 1px solid #e0e0e0; }
    .streamlit-expanderHeader { background-color: #f8f9fa; border-radius: 6px; }
    .stSelectbox > div > div, .stMultiSelect > div > div { border-radius: 6px; }
    hr { border: none; height: 1px; background-color: #e0e0e0; margin: 1.5rem 0; }
</style>
""", unsafe_allow_html=True)


# ==================== FONCTIONS UTILITAIRES (PRINCE) ====================

def safe_column_coordinates(model, df):
    try:
        coords = getattr(model, 'column_coordinates_', None)
        if coords is not None and isinstance(coords, pd.DataFrame) and not coords.empty:
            return coords
    except Exception:
        pass
    try:
        coords = model.column_coordinates(df)
        if coords is not None and isinstance(coords, pd.DataFrame) and not coords.empty:
            return coords
    except Exception:
        pass
    try:
        if hasattr(model, 'V_') and hasattr(model, 'eigenvalues_'):
            V = model.V_
            eigenvalues = model.eigenvalues_
            coords = pd.DataFrame(
                V.T * np.sqrt(eigenvalues),
                index=df.columns if hasattr(df, 'columns') else range(V.shape[1]),
                columns=[f"component {i}" for i in range(len(eigenvalues))]
            )
            return coords
    except Exception:
        pass
    return None


def safe_row_coordinates(model, df):
    try:
        return model.row_coordinates(df)
    except Exception:
        pass
    try:
        coords = getattr(model, 'row_coordinates_', None)
        if coords is not None:
            return coords
    except Exception:
        pass
    try:
        return model.transform(df)
    except Exception as e:
        raise RuntimeError(f"Impossible d'obtenir les coordonnées des individus : {e}")


def safe_contributions(model, df=None):
    try:
        contrib = getattr(model, 'column_contributions_', None)
        if contrib is not None and isinstance(contrib, pd.DataFrame) and not contrib.empty:
            return contrib
    except Exception:
        pass
    try:
        if df is not None:
            contrib = model.column_contributions(df)
            if contrib is not None:
                return contrib
    except Exception:
        pass
    return None


def safe_eigenvalues(model):
    try:
        ev = getattr(model, 'eigenvalues_', None)
        if ev is not None:
            return ev
    except Exception:
        pass
    try:
        summary = model.eigenvalues_summary
        return summary.iloc[:, 0].values
    except Exception:
        pass
    return None


# ==================== CHARGEMENT & PREPROCESSING ====================

@st.cache_data
def load_data(uploaded_file=None):
    df = None
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        local_paths = [
            'data/WA_Fn-UseC_-HR-Employee-Attrition.csv',
            'WA_Fn-UseC_-HR-Employee-Attrition.csv',
            '/mnt/user-data/uploads/WA_Fn-UseC_-HR-Employee-Attrition.csv'
        ]
        for path in local_paths:
            try:
                df = pd.read_csv(path)
                break
            except Exception:
                continue

    if df is not None:
        numeric_columns = [
            'Age', 'DailyRate', 'DistanceFromHome', 'Education',
            'EnvironmentSatisfaction', 'HourlyRate', 'JobInvolvement',
            'JobLevel', 'JobSatisfaction', 'MonthlyIncome', 'MonthlyRate',
            'NumCompaniesWorked', 'PercentSalaryHike', 'PerformanceRating',
            'RelationshipSatisfaction', 'StockOptionLevel', 'TotalWorkingYears',
            'TrainingTimesLastYear', 'WorkLifeBalance', 'YearsAtCompany',
            'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager'
        ]
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        return df
    return None


def get_variable_types(df):
    quant_vars = df.select_dtypes(include=[np.number]).columns.tolist()
    qual_vars = df.select_dtypes(include=['object']).columns.tolist()
    return quant_vars, qual_vars


def preprocess_for_analysis(df):
    df_processed = df.copy()
    cols_to_drop = ['EmployeeCount', 'StandardHours', 'Over18', 'EmployeeNumber']
    cols_to_drop = [col for col in cols_to_drop if col in df_processed.columns]
    df_processed = df_processed.drop(columns=cols_to_drop)
    for col in df_processed.columns:
        if df_processed[col].dtype == 'object':
            try:
                converted = pd.to_numeric(df_processed[col], errors='coerce')
                if converted.notna().mean() > 0.9:
                    df_processed[col] = converted
            except Exception:
                pass
    return df_processed


def encode_categorical(df, target_col='Attrition'):
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
        st.markdown('<div class="metric-card"><h3>📊 Analyse Factorielle</h3><p>ACP, ACM, AFDM, AFC</p></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card"><h3>🔮 Clustering</h3><p>K-Means & CAH</p></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card"><h3>🤖 Prédiction</h3><p>Classification ML</p></div>', unsafe_allow_html=True)

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

    st.markdown("### 📦 Justification du Choix du Dataset")
    st.markdown("""
    <div class="success-box">
    <strong>Pourquoi ce dataset ?</strong><br><br>
    Le dataset <em>IBM HR Analytics Employee Attrition & Performance</em> a été choisi pour les raisons suivantes :
    <ol>
    <li><strong>Données mixtes (quantitatives + qualitatives) :</strong> Le dataset contient à la fois des variables 
    numériques (Age, MonthlyIncome, YearsAtCompany…) et catégorielles (Department, JobRole, OverTime…), 
    ce qui le rend particulièrement adapté à l'<strong>AFDM</strong> et permet aussi de comparer avec l'ACP, l'ACM et l'AFC.</li>
    <li><strong>Problématique métier concrète :</strong> L'attrition (départ volontaire des employés) est un enjeu 
    stratégique majeur en Ressources Humaines. Ce dataset permet de poser une problématique réelle et interprétable.</li>
    <li><strong>Taille et qualité :</strong> Avec 1 470 observations et 35 variables, le dataset est suffisamment 
    riche pour les analyses factorielles (nombre d'individus >> nombre de variables), sans valeurs manquantes.</li>
    <li><strong>Variable cible binaire :</strong> La variable <code>Attrition</code> (Yes/No) sert naturellement 
    de <strong>variable supplémentaire (illustrative)</strong> pour valider les axes factoriels, 
    et de variable cible pour la classification supervisée.</li>
    <li><strong>Déséquilibre réaliste :</strong> Le taux d'attrition de 16,1% reflète la réalité du monde du travail, 
    ce qui rend l'analyse plus pertinente et met en évidence la nécessité de techniques comme SMOTE.</li>
    </ol>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🎯 Problématique")
    st.markdown("""
    <div class="info-box">
    <strong>Question principale :</strong> Quels sont les facteurs déterminants qui influencent la décision d'un employé 
    de quitter l'entreprise, et comment peut-on prédire et prévenir l'attrition ?<br><br>
    <strong>Sous-questions :</strong>
    <ul>
    <li>Quelles variables (quantitatives et qualitatives) discriminent le plus les employés qui partent vs restent ?</li>
    <li>Existe-t-il des profils types d'employés à risque identifiables par analyse factorielle ?</li>
    <li>Comment les plans factoriels révèlent-ils la structure sous-jacente des données RH ?</li>
    <li>Peut-on valider l'analyse en projetant la variable Attrition comme point supplémentaire ?</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

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
        - Random Forest, Gradient Boosting, Logistic Regression
        - Gestion du déséquilibre (SMOTE)
        - Validation croisée
        - Importance des variables
        """)

    st.markdown("### 📁 Description du Dataset")
    st.write("""
    Le dataset **IBM HR Analytics Employee Attrition & Performance** contient 1470 observations 
    et 35 variables décrivant les caractéristiques des employés.
    """)

    with st.expander("📊 Détail des variables du dataset"):
        st.markdown("""
        | Type | Variables | Exemples |
        |------|----------|----------|
        | **Quantitatives continues** | 11 variables | Age, MonthlyIncome, DailyRate, DistanceFromHome, YearsAtCompany… |
        | **Quantitatives ordinales** | 10 variables | Education (1-5), JobSatisfaction (1-4), WorkLifeBalance (1-4)… |
        | **Qualitatives nominales** | 7 variables | Department, JobRole, MaritalStatus, OverTime, Gender… |
        | **Qualitatives binaires** | 2 variables | Attrition (Yes/No), OverTime (Yes/No) |
        | **Constantes (supprimées)** | 4 variables | EmployeeCount, StandardHours, Over18, EmployeeNumber |
        
        **Variable cible (supplémentaire) :** `Attrition` — utilisée comme variable illustrative en analyse factorielle 
        pour valider l'interprétation des axes, puis comme variable à prédire en classification.
        """)


# ==================== PAGE: EXPLORATION ====================

def page_exploration(df):
    st.markdown("## 📊 Exploration des Données (EDA)")
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
        st.markdown("#### Distribution de la Variable Cible (Attrition)")
        col1, col2 = st.columns(2)
        with col1:
            attrition_counts = df['Attrition'].value_counts()
            fig = px.pie(values=attrition_counts.values, names=attrition_counts.index,
                        title="Répartition de l'Attrition",
                        color_discrete_sequence=['#10B981', '#EF4444'])
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.markdown("""
            <div class="warning-box">
            <strong>⚠️ Déséquilibre des classes:</strong><br>
            Le dataset présente un déséquilibre significatif avec environ 16% d'attrition.
            </div>
            """, unsafe_allow_html=True)

        st.markdown("#### Distribution des Variables Quantitatives")
        selected_quant = st.multiselect("Sélectionner les variables:", quant_vars, default=quant_vars[:4])
        if selected_quant:
            n_cols = 2
            n_rows = (len(selected_quant) + 1) // 2
            fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=selected_quant)
            for i, var in enumerate(selected_quant):
                row = i // n_cols + 1
                col = i % n_cols + 1
                fig.add_trace(go.Histogram(x=df[var], name=var, marker_color='#667eea'), row=row, col=col)
            fig.update_layout(height=300*n_rows, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Distribution des Variables Qualitatives")
        selected_qual = st.selectbox("Sélectionner une variable:", qual_vars)
        fig = px.histogram(df, x=selected_qual, color='Attrition', barmode='group',
                          color_discrete_map={'No': '#10B981', 'Yes': '#EF4444'},
                          title=f"Distribution de {selected_qual} par Attrition")
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.markdown("### Analyse des Corrélations")
        df_encoded, _ = encode_categorical(df)
        corr_matrix = df_encoded.corr()
        attrition_corr = corr_matrix['Attrition'].drop('Attrition').sort_values(key=abs, ascending=False)
        fig = px.bar(x=attrition_corr.head(15).values, y=attrition_corr.head(15).index,
                    orientation='h', color=attrition_corr.head(15).values,
                    color_continuous_scale='RdBu_r',
                    title="Top 15 Corrélations avec l'Attrition")
        fig.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig, use_container_width=True)

        vars_for_heatmap = st.multiselect("Variables pour la heatmap:",
                                          corr_matrix.columns.tolist(),
                                          default=corr_matrix.columns.tolist()[:15])
        if vars_for_heatmap:
            fig = px.imshow(corr_matrix.loc[vars_for_heatmap, vars_for_heatmap],
                           color_continuous_scale='RdBu_r', aspect='auto',
                           title="Matrice de Corrélation")
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)

    with tab4:
        st.markdown("### Détection des Outliers")
        selected_vars = st.multiselect("Sélectionner les variables:", quant_vars,
                                       default=['Age', 'MonthlyIncome', 'YearsAtCompany', 'TotalWorkingYears'])
        if selected_vars:
            fig = make_subplots(rows=1, cols=len(selected_vars), subplot_titles=selected_vars)
            for i, var in enumerate(selected_vars):
                fig.add_trace(go.Box(y=df[var], name=var, marker_color='#667eea'), row=1, col=i+1)
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        outlier_counts = {}
        for var in quant_vars:
            Q1 = df[var].quantile(0.25)
            Q3 = df[var].quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((df[var] < Q1 - 1.5*IQR) | (df[var] > Q3 + 1.5*IQR)).sum()
            outlier_counts[var] = outliers
        outlier_df = pd.DataFrame.from_dict(outlier_counts, orient='index', columns=['Outliers'])
        outlier_df = outlier_df.sort_values('Outliers', ascending=False)
        fig = px.bar(outlier_df.head(10), x=outlier_df.head(10).index, y='Outliers',
                    title="Top 10 Variables avec Outliers", color='Outliers', color_continuous_scale='Reds')
        st.plotly_chart(fig, use_container_width=True)


# ==================== PAGE: ANALYSE FACTORIELLE ====================

def page_afdm(df):
    st.markdown("## 🔬 Analyse Factorielle")

    df_processed = preprocess_for_analysis(df)
    numeric_cols = df_processed.select_dtypes(include=[np.number]).columns.tolist()
    object_cols = df_processed.select_dtypes(include=['object', 'category']).columns.tolist()

    quant_vars = [col for col in numeric_cols if df_processed[col].nunique() > 10]
    qual_vars = object_cols + [col for col in numeric_cols if df_processed[col].nunique() <= 10]
    qual_vars_active = [v for v in qual_vars if v != 'Attrition']
    quant_vars = [v for v in quant_vars if v != 'Attrition']

    st.markdown("### 🎯 Choix du Modèle d'Analyse")
    col1, col2 = st.columns([1, 2])

    with col1:
        model_type = st.selectbox(
            "Sélectionnez le type d'analyse:",
            ["AFDM (Données Mixtes)", "ACP (Quantitatives)", "ACM (Qualitatives)", "AFC (Tableau de Contingence)"],
        )

    with col2:
        descriptions = {
            "AFDM (Données Mixtes)": "<strong>AFDM</strong> — Combine ACP et ACM. Adapté aux données avec variables quantitatives ET qualitatives.",
            "ACP (Quantitatives)": "<strong>ACP</strong> — Réduit la dimensionnalité sur variables quantitatives uniquement en préservant la variance.",
            "ACM (Qualitatives)": "<strong>ACM</strong> — Analyse les associations entre modalités de variables qualitatives uniquement.",
            "AFC (Tableau de Contingence)": "<strong>AFC</strong> — Analyse la relation entre exactement 2 variables qualitatives via un tableau de contingence.",
        }
        st.markdown(f'<div class="info-box">{descriptions[model_type]}</div>', unsafe_allow_html=True)

    # Justification détaillée du choix
    justifications = {
        "AFDM (Données Mixtes)": """
        <div class="success-box">
        <strong>Pourquoi l'AFDM est le modèle principal recommandé ?</strong><br><br>
        Le dataset IBM HR contient simultanément des variables <strong>quantitatives</strong> (Age, MonthlyIncome, 
        YearsAtCompany…) et <strong>qualitatives</strong> (Department, JobRole, OverTime…). 
        L'AFDM (Analyse Factorielle des Données Mixtes) est la seule méthode qui traite les deux types 
        de variables dans un cadre unifié, en normalisant les contributions de chaque type.<br><br>
        <strong>Avantages :</strong> Vue d'ensemble complète, pas de perte d'information par encodage forcé, 
        pondération équilibrée entre variables numériques et catégorielles.<br>
        <strong>Outil :</strong> <code>prince.FAMD</code> (Python) — implémentation de l'AFDM de Pagès (2004).
        </div>
        """,
        "ACP (Quantitatives)": """
        <div class="success-box">
        <strong>Pourquoi proposer l'ACP en complément ?</strong><br><br>
        L'ACP permet d'analyser uniquement les variables numériques (Age, MonthlyIncome, YearsAtCompany, 
        DistanceFromHome…) en identifiant les axes de plus grande variance. Elle est utile pour :<br>
        • Visualiser les corrélations entre variables quantitatives (cercle des corrélations)<br>
        • Identifier les dimensions principales de variation des profils d'employés<br>
        • L'<strong>ACP centrée-réduite</strong> est recommandée ici car les variables ont des échelles très 
        différentes (Age en années vs MonthlyIncome en dollars).<br><br>
        <strong>Outil :</strong> <code>prince.PCA</code> — avec option centrée ou centrée-réduite.
        </div>
        """,
        "ACM (Qualitatives)": """
        <div class="success-box">
        <strong>Pourquoi proposer l'ACM en complément ?</strong><br><br>
        L'ACM est adaptée pour explorer les associations entre les variables qualitatives du dataset : 
        Department, JobRole, MaritalStatus, BusinessTravel, OverTime, ainsi que les variables ordinales 
        traitées comme catégorielles (Education, JobSatisfaction, WorkLifeBalance…).<br>
        • Elle révèle quelles <strong>modalités</strong> sont fréquemment associées chez les mêmes individus<br>
        • La correction de <strong>Benzécri</strong> est recommandée pour compenser la sous-estimation 
        structurelle des valeurs propres en ACM.<br><br>
        <strong>Outil :</strong> <code>prince.MCA</code> — avec corrections Benzécri/Greenacre.
        </div>
        """,
        "AFC (Tableau de Contingence)": """
        <div class="success-box">
        <strong>Pourquoi proposer l'AFC ?</strong><br><br>
        L'AFC (Analyse Factorielle des Correspondances) permet d'étudier la relation entre 
        <strong>exactement deux variables qualitatives</strong> via leur tableau de contingence. 
        Elle est utile ici pour explorer des associations spécifiques, par exemple :<br>
        • <strong>Department × JobRole</strong> : structure organisationnelle<br>
        • <strong>OverTime × Attrition</strong> : lien heures supplémentaires / départs<br>
        • Le test du χ² valide si la relation est statistiquement significative avant d'interpréter le biplot.<br><br>
        <strong>Outil :</strong> <code>prince.CA</code> — avec test d'indépendance intégré.
        </div>
        """,
    }
    with st.expander("📖 Justification détaillée du choix de ce modèle", expanded=False):
        st.markdown(justifications[model_type], unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### ⚙️ Paramètres de l'Analyse")
    col1, col2 = st.columns(2)

    with col1:
        n_components = st.slider("Nombre de composantes:", 2, 10, 5)

    with col2:
        acp_type = None
        acm_correction = None
        if model_type == "ACP (Quantitatives)":
            acp_type = st.radio("Type d'ACP:", ["Centrée-Réduite", "Centrée uniquement"])
        elif model_type == "ACM (Qualitatives)":
            acm_correction = st.radio(
                "Correction des valeurs propres:",
                ["Benzécri", "Greenacre", "Aucune"],
                help="En ACM les valeurs propres sont structurellement sous-estimées. Benzécri et Greenacre corrigent ce biais."
            )

    with st.expander("🔧 Sélection des Variables"):
        if model_type == "ACP (Quantitatives)":
            selected_quant = st.multiselect("Variables quantitatives:", quant_vars, default=quant_vars)
            selected_qual = []
            afc_var1, afc_var2 = None, None
        elif model_type == "ACM (Qualitatives)":
            selected_qual = st.multiselect("Variables qualitatives:", qual_vars_active, default=qual_vars_active[:10])
            selected_quant = []
            afc_var1, afc_var2 = None, None
        elif model_type == "AFC (Tableau de Contingence)":
            afc_var1 = st.selectbox("Variable 1 (lignes):", qual_vars_active, index=0)
            remaining = [v for v in qual_vars_active if v != afc_var1]
            afc_var2 = st.selectbox("Variable 2 (colonnes):", remaining, index=0)
            selected_quant = []
            selected_qual = [afc_var1, afc_var2]
        else:
            selected_quant = st.multiselect("Variables quantitatives:", quant_vars, default=quant_vars)
            selected_qual = st.multiselect("Variables qualitatives:", qual_vars_active, default=qual_vars_active[:5])
            afc_var1, afc_var2 = None, None

    if st.button(f"🚀 Lancer l'Analyse ({model_type.split(' ')[0]})", type="primary"):
        if model_type == "AFDM (Données Mixtes)" and (len(selected_quant) == 0 or len(selected_qual) == 0):
            st.error("L'AFDM nécessite au moins une variable quantitative ET une variable qualitative.")
            return
        if model_type == "ACP (Quantitatives)" and len(selected_quant) < 2:
            st.error("L'ACP nécessite au moins 2 variables quantitatives.")
            return
        if model_type == "ACM (Qualitatives)" and len(selected_qual) < 2:
            st.error("L'ACM nécessite au moins 2 variables qualitatives.")
            return

        with st.spinner(f"Calcul en cours..."):
            try:
                row_coords = None
                col_coords = None
                df_analysis = None
                model = None
                analysis_name = ""
                contingency_table = None
                has_attrition = False
                is_normalized_pca = False

                # ==================== ACP ====================
                if model_type == "ACP (Quantitatives)":
                    analysis_name = "ACP"
                    df_analysis = df_processed[selected_quant].dropna().copy()
                    df_analysis = df_analysis.astype(float)

                    is_normalized_pca = (acp_type == "Centrée-Réduite")
                    model = prince.PCA(
                        n_components=n_components,
                        rescale_with_mean=True,
                        rescale_with_std=is_normalized_pca,
                        random_state=42
                    )
                    model.fit(df_analysis)

                    row_coords = safe_row_coordinates(model, df_analysis)
                    col_coords = safe_column_coordinates(model, df_analysis)

                    # Toujours recalculer les corrélations variables-composantes
                    # pour le cercle des corrélations (garantit des valeurs dans [-1, 1])
                    if col_coords is not None:
                        corr_cols = []
                        for j in range(row_coords.shape[1]):
                            corr_col = []
                            for var in df_analysis.columns:
                                r = np.corrcoef(df_analysis[var].values, row_coords.iloc[:, j].values)[0, 1]
                                corr_col.append(r)
                            corr_cols.append(corr_col)
                        col_coords = pd.DataFrame(
                            np.array(corr_cols).T,
                            index=df_analysis.columns,
                            columns=[f"component {i}" for i in range(len(corr_cols))]
                        )

                    attrition_index = df_processed.loc[df_analysis.index, 'Attrition'] if 'Attrition' in df_processed.columns else None
                    if attrition_index is not None:
                        row_coords = row_coords.copy()
                        row_coords['Attrition'] = attrition_index.values
                        has_attrition = True

                # ==================== ACM ====================
                elif model_type == "ACM (Qualitatives)":
                    analysis_name = "ACM"
                    df_analysis = df_processed[selected_qual].dropna().copy()

                    for col in selected_qual:
                        if df_analysis[col].nunique() < 2:
                            st.error(f"La variable '{col}' n'a qu'une seule modalité après nettoyage.")
                            return
                        df_analysis[col] = df_analysis[col].astype(str)

                    correction_map = {"Benzécri": "benzecri", "Greenacre": "greenacre", "Aucune": None}
                    correction_value = correction_map.get(acm_correction, "benzecri")

                    model = prince.MCA(
                        n_components=n_components,
                        correction=correction_value,
                        random_state=42
                    )
                    model.fit(df_analysis)

                    row_coords = safe_row_coordinates(model, df_analysis)
                    col_coords = safe_column_coordinates(model, df_analysis)

                    attrition_index = df_processed.loc[df_analysis.index, 'Attrition'] if 'Attrition' in df_processed.columns else None
                    if attrition_index is not None:
                        row_coords = row_coords.copy()
                        row_coords['Attrition'] = attrition_index.values
                        has_attrition = True

                # ==================== AFC ====================
                elif model_type == "AFC (Tableau de Contingence)":
                    analysis_name = "AFC"
                    df_afc = df_processed[[afc_var1, afc_var2]].dropna()
                    contingency_table = pd.crosstab(df_afc[afc_var1], df_afc[afc_var2])

                    n_rows_ct, n_cols_ct = contingency_table.shape
                    max_components = min(n_components, min(n_rows_ct, n_cols_ct) - 1)

                    if max_components < 1:
                        st.error("Les variables sélectionnées ont trop peu de modalités pour l'AFC.")
                        return

                    model = prince.CA(n_components=max_components, random_state=42)
                    model.fit(contingency_table)

                    row_coords = safe_row_coordinates(model, contingency_table)
                    col_coords = safe_column_coordinates(model, contingency_table)

                    has_attrition = False
                    df_analysis = contingency_table

                # ==================== AFDM ====================
                else:
                    analysis_name = "AFDM"
                    data_dict = {}
                    for col in selected_quant:
                        values = pd.to_numeric(df_processed[col], errors='coerce')
                        data_dict[col] = values.astype('float64')
                    for col in selected_qual:
                        data_dict[col] = df_processed[col].astype(str)

                    df_analysis = pd.DataFrame(data_dict).dropna()

                    if len(df_analysis) == 0:
                        st.error("Aucune donnée valide après nettoyage.")
                        return

                    for c in selected_quant:
                        if c in df_analysis.columns:
                            df_analysis[c] = df_analysis[c].astype('float64')
                    for c in selected_qual:
                        if c in df_analysis.columns:
                            df_analysis[c] = df_analysis[c].astype(str)

                    numeric_detected = df_analysis.select_dtypes(include=['float']).columns.tolist()
                    object_detected = df_analysis.select_dtypes(include=['object']).columns.tolist()

                    st.info(f"Variables détectées — Numériques (float): {len(numeric_detected)}, Catégorielles (str): {len(object_detected)}")

                    if len(numeric_detected) == 0:
                        st.error("Aucune variable numérique détectée. Utilisez l'ACM à la place.")
                        return
                    if len(object_detected) == 0:
                        st.error("Aucune variable catégorielle détectée. Utilisez l'ACP à la place.")
                        return

                    model = prince.FAMD(n_components=n_components, n_iter=10, random_state=42)
                    model.fit(df_analysis)

                    row_coords = safe_row_coordinates(model, df_analysis)
                    col_coords = safe_column_coordinates(model, df_analysis)

                    attrition_vals = df_processed.loc[df_analysis.index, 'Attrition'] if 'Attrition' in df_processed.columns else None
                    if attrition_vals is not None:
                        row_coords = row_coords.copy()
                        row_coords['Attrition'] = attrition_vals.values
                        has_attrition = True

                # ==================== AFFICHAGE ====================
                st.markdown(f"### 📊 Résultats de l'{analysis_name}")
                tab1, tab2, tab3, tab4 = st.tabs(["📉 Inertie", "👥 Individus", "📊 Variables", "🎯 Interprétation"])

                with tab1:
                    display_eigenvalues(model, n_components, analysis_name,
                                        is_normalized_pca=is_normalized_pca,
                                        contingency_table=contingency_table)
                with tab2:
                    if analysis_name == "AFC":
                        display_individuals_afc(model, row_coords, col_coords, afc_var1, afc_var2, contingency_table)
                    else:
                        display_individuals(row_coords, model, n_components, analysis_name, has_attrition)
                with tab3:
                    if analysis_name == "AFC":
                        display_variables_afc(model, row_coords, col_coords, afc_var1, afc_var2, contingency_table)
                    else:
                        display_variables(model, col_coords, n_components, analysis_name, df_analysis)
                with tab4:
                    display_interpretation(model, col_coords, n_components, analysis_name, df_analysis)

                # ==================== POINTS SUPPLÉMENTAIRES ====================
                if analysis_name in ["ACP", "ACM", "AFDM"] and has_attrition:
                    st.markdown("---")
                    st.markdown("### 🔍 Validation par Points Supplémentaires")
                    st.markdown("""
                    <div class="info-box">
                    <strong>Principe :</strong> La variable <code>Attrition</code> n'est <strong>pas utilisée 
                    comme variable active</strong> dans l'analyse factorielle. Elle est projetée 
                    <em>a posteriori</em> sur les plans factoriels en tant que <strong>variable supplémentaire 
                    (illustrative)</strong> pour vérifier si les axes factoriels permettent de discriminer 
                    les employés partants des restants. Si les barycentres des deux groupes (Yes/No) 
                    sont bien séparés, cela <strong>valide l'interprétation</strong> des axes.
                    </div>
                    """, unsafe_allow_html=True)

                    coord_cols_sup = [col for col in row_coords.columns if col != 'Attrition']
                    rename_sup = {old: f"F{i}" for i, old in enumerate(coord_cols_sup)}
                    rc_sup = row_coords.rename(columns=rename_sup)

                    # Barycentres par groupe Attrition
                    barycentres = rc_sup.groupby('Attrition')[[f'F{i}' for i in range(min(3, len(coord_cols_sup)))]].mean()
                    st.markdown("#### Barycentres des groupes Attrition (points supplémentaires)")
                    st.dataframe(barycentres.round(4), use_container_width=True)

                    # Test de significativité (t-test sur chaque axe)
                    group_yes = rc_sup[rc_sup['Attrition'] == 'Yes']
                    group_no = rc_sup[rc_sup['Attrition'] == 'No']
                    test_results = []
                    for i in range(min(3, len(coord_cols_sup))):
                        axis_name = f'F{i}'
                        t_stat, p_val = stats.ttest_ind(group_yes[axis_name].dropna(), group_no[axis_name].dropna())
                        test_results.append({
                            'Axe': axis_name,
                            'Barycentre Yes': barycentres.loc['Yes', axis_name] if 'Yes' in barycentres.index else np.nan,
                            'Barycentre No': barycentres.loc['No', axis_name] if 'No' in barycentres.index else np.nan,
                            't-stat': t_stat,
                            'p-value': p_val,
                            'Significatif (p<0.05)': '✅ Oui' if p_val < 0.05 else '❌ Non'
                        })
                    test_df = pd.DataFrame(test_results)
                    st.dataframe(test_df.round(4), use_container_width=True)

                    # Visualisation des barycentres sur le plan F0-F1
                    fig_sup = px.scatter(rc_sup, x='F0', y='F1', color='Attrition',
                                        color_discrete_map={'No': '#10B981', 'Yes': '#EF4444'},
                                        title="Plan F0-F1 avec barycentres des groupes Attrition (★)",
                                        opacity=0.3)
                    # Ajouter les barycentres comme étoiles
                    for grp in barycentres.index:
                        color_grp = '#EF4444' if grp == 'Yes' else '#10B981'
                        fig_sup.add_trace(go.Scatter(
                            x=[barycentres.loc[grp, 'F0']], y=[barycentres.loc[grp, 'F1']],
                            mode='markers+text', text=[f'★ {grp}'], textposition='top center',
                            marker=dict(size=20, color=color_grp, symbol='star'),
                            name=f'Barycentre {grp}', showlegend=True
                        ))
                    fig_sup.add_hline(y=0, line_dash="dash", line_color="gray")
                    fig_sup.add_vline(x=0, line_dash="dash", line_color="gray")
                    fig_sup.update_layout(height=600)
                    st.plotly_chart(fig_sup, use_container_width=True)

                    # Interprétation
                    significant_axes = [r['Axe'] for r in test_results if r['p-value'] < 0.05]
                    if significant_axes:
                        st.markdown(f"""
                        <div class="success-box">
                        <strong>✅ Validation réussie :</strong> La variable supplémentaire Attrition est 
                        significativement discriminée sur les axes <strong>{', '.join(significant_axes)}</strong> 
                        (test t, p < 0.05). Cela confirme que les dimensions factorielles captent bien 
                        les facteurs liés au départ des employés.
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div class="warning-box">
                        <strong>⚠️ Résultat mitigé :</strong> Les barycentres des groupes Attrition ne sont 
                        significativement séparés sur aucun des premiers axes. L'attrition est un phénomène 
                        multifactoriel qui n'est pas entièrement capté par les premières dimensions.
                        </div>
                        """, unsafe_allow_html=True)

                st.session_state['factor_coords'] = row_coords
                st.session_state['factor_model'] = model
                st.session_state['analysis_name'] = analysis_name
                st.success(f"✅ {analysis_name} terminée avec succès!")

            except Exception as e:
                st.error(f"Erreur lors de l'analyse : {str(e)}")
                import traceback
                with st.expander("Détails de l'erreur"):
                    st.code(traceback.format_exc())


# ==================== AFFICHAGE: VALEURS PROPRES ====================

def display_eigenvalues(model, n_components, analysis_name,
                        is_normalized_pca=False, contingency_table=None):
    st.markdown("#### Valeurs Propres et Inertie Expliquée")

    # AFC: Test du Chi-2 d'indépendance
    if analysis_name == "AFC" and contingency_table is not None:
        chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
        st.markdown(f"""
        <div class="{'success-box' if p_value < 0.05 else 'warning-box'}">
        <strong>Test du χ² d'indépendance :</strong><br>
        • χ² = {chi2:.2f}, ddl = {dof}, p-value = {p_value:.2e}<br>
        • <strong>{'Les variables sont significativement liées (p < 0.05) → l\'AFC est pertinente.' 
           if p_value < 0.05 
           else '⚠️ Les variables ne sont pas significativement liées (p ≥ 0.05) → l\'AFC est peu pertinente.'}</strong>
        </div>
        """, unsafe_allow_html=True)

    try:
        eigenvalues = model.eigenvalues_summary
    except Exception:
        st.warning("Résumé des valeurs propres non disponible pour ce modèle.")
        return

    col1, col2 = st.columns(2)
    with col1:
        st.dataframe(eigenvalues.round(4), use_container_width=True)

    with col2:
        pct_col = None
        for candidate in ['% of variance', '% of variance explained', 'Percentage of variance']:
            if candidate in eigenvalues.columns:
                pct_col = candidate
                break
        if pct_col is None and eigenvalues.shape[1] >= 2:
            pct_col = eigenvalues.columns[1]

        if pct_col:
            pct_values = eigenvalues[pct_col].values
            if len(pct_values) > 0 and isinstance(pct_values[0], str):
                pct_values = np.array([float(v.strip('%')) for v in pct_values])

            fig = go.Figure()
            fig.add_trace(go.Bar(x=[f"F{i+1}" for i in range(len(pct_values))], y=pct_values,
                                name='% Variance', marker_color='#667eea'))
            fig.add_trace(go.Scatter(x=[f"F{i+1}" for i in range(len(pct_values))], y=np.cumsum(pct_values),
                                    name='% Cumulé', mode='lines+markers', marker_color='#EF4444'))
            fig.update_layout(title="Éboulis des Valeurs Propres",
                             xaxis_title="Composantes", yaxis_title="% de Variance")
            st.plotly_chart(fig, use_container_width=True)

    # Critères d'interprétation spécifiques à chaque méthode
    if analysis_name == "ACP":
        eigenvalues_arr = safe_eigenvalues(model)
        if eigenvalues_arr is not None:
            if is_normalized_pca:
                kaiser_count = int((np.array(eigenvalues_arr) > 1).sum())
                st.markdown(f"""
                <div class="success-box">
                <strong>Critères de sélection des axes (ACP centrée-réduite) :</strong><br>
                • <strong>Règle de Kaiser :</strong> {kaiser_count} axe(s) avec valeur propre &gt; 1<br>
                • <strong>Critère du coude :</strong> Observer l'inflexion de la courbe<br>
                • <strong>Inertie cumulée :</strong> Retenir les axes expliquant 70–80% de l'inertie
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="warning-box">
                <strong>ACP centrée uniquement (non réduite) :</strong><br>
                • La règle de Kaiser (λ &gt; 1) <strong>ne s'applique pas</strong> car les variables ne sont pas standardisées.<br>
                • Utilisez le <strong>critère du coude</strong> ou le seuil d'<strong>inertie cumulée (70–80%)</strong>.
                </div>
                """, unsafe_allow_html=True)

    elif analysis_name == "ACM":
        st.markdown("""
        <div class="success-box">
        <strong>Interprétation (ACM) :</strong><br>
        • En ACM, les valeurs propres sont naturellement faibles (sous-estimation structurelle).<br>
        • La correction de Benzécri/Greenacre ajuste ces valeurs pour mieux refléter la variance réelle.<br>
        • Retenir les axes au coude de la courbe ou expliquant la majorité de l'inertie corrigée.
        </div>
        """, unsafe_allow_html=True)

    elif analysis_name == "AFC":
        total_inertia = None
        try:
            total_inertia = getattr(model, 'total_inertia_', None)
        except Exception:
            pass
        inertia_text = f"<br>• <strong>Inertie totale :</strong> {total_inertia:.4f}" if total_inertia else ""
        st.markdown(f"""
        <div class="success-box">
        <strong>Interprétation (AFC) :</strong>{inertia_text}<br>
        • L'inertie totale mesure l'écart à l'indépendance entre les deux variables.<br>
        • Plus elle est élevée, plus l'association entre les variables est forte.
        </div>
        """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div class="success-box">
        <strong>Interprétation (AFDM) :</strong> Retenir les axes au niveau du coude de la courbe 
        ou expliquant au moins 70–80% de l'inertie totale.
        </div>
        """, unsafe_allow_html=True)


# ==================== AFFICHAGE: INDIVIDUS ====================

def display_individuals(row_coords, model, n_components, analysis_name, has_attrition):
    st.markdown("#### Projection des Individus")

    coord_cols = [col for col in row_coords.columns if col != 'Attrition']
    n_dims = len(coord_cols)
    if n_dims < 2:
        st.warning("Pas assez de dimensions pour afficher la projection.")
        return

    rename_dict = {old: f"F{i}" for i, old in enumerate(coord_cols)}
    row_coords_plot = row_coords.rename(columns=rename_dict)

    col1, col2 = st.columns(2)
    with col1:
        axis_x = st.selectbox("Axe X:", [f"F{i}" for i in range(n_dims)], index=0, key="ind_x")
    with col2:
        axis_y = st.selectbox("Axe Y:", [f"F{i}" for i in range(n_dims)], index=min(1, n_dims-1), key="ind_y")

    if has_attrition and 'Attrition' in row_coords_plot.columns:
        fig = px.scatter(row_coords_plot, x=axis_x, y=axis_y, color='Attrition',
                        color_discrete_map={'No': '#10B981', 'Yes': '#EF4444'},
                        title=f"Projection des Individus — Plan {axis_x}-{axis_y}", opacity=0.6)
    else:
        fig = px.scatter(row_coords_plot, x=axis_x, y=axis_y,
                        title=f"Projection des Individus — Plan {axis_x}-{axis_y}", opacity=0.6)

    try:
        eigenvalues = model.eigenvalues_summary
        pct_col = [c for c in eigenvalues.columns if 'variance' in c.lower()]
        if pct_col:
            x_idx, y_idx = int(axis_x[1]), int(axis_y[1])
            if x_idx < len(eigenvalues) and y_idx < len(eigenvalues):
                x_var = eigenvalues[pct_col[0]].iloc[x_idx]
                y_var = eigenvalues[pct_col[0]].iloc[y_idx]
                if isinstance(x_var, str):
                    x_var, y_var = float(x_var.strip('%')), float(y_var.strip('%'))
                fig.update_layout(xaxis_title=f"{axis_x} ({x_var:.2f}%)", yaxis_title=f"{axis_y} ({y_var:.2f}%)")
    except Exception:
        pass

    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.add_vline(x=0, line_dash="dash", line_color="gray")
    st.plotly_chart(fig, use_container_width=True)

    interpretations = {
        "ACP": "Les individus proches ont des profils quantitatifs similaires. La coloration par Attrition permet d'identifier visuellement les groupes à risque.",
        "ACM": "Les individus proches partagent les mêmes modalités qualitatives. Deux individus superposés ont des réponses identiques.",
        "AFDM": "Les individus proches ont des profils similaires (quantitatif + qualitatif). La coloration par Attrition montre la séparation entre partants et restants.",
    }
    msg = interpretations.get(analysis_name, "Les individus proches ont des profils similaires.")
    st.markdown(f'<div class="info-box"><strong>Lecture ({analysis_name}) :</strong> {msg}</div>', unsafe_allow_html=True)


def display_individuals_afc(model, row_coords, col_coords, var1, var2, contingency_table):
    st.markdown("#### Biplot AFC — Modalités dans le Plan Factoriel")

    row_plot = row_coords.copy()
    col_plot = col_coords.copy()
    row_plot.columns = [f"F{i}" for i in range(row_plot.shape[1])]
    col_plot.columns = [f"F{i}" for i in range(col_plot.shape[1])]
    n_dims = row_plot.shape[1]

    col1_ui, col2_ui = st.columns(2)
    with col1_ui:
        axis_x = st.selectbox("Axe X:", [f"F{i}" for i in range(n_dims)], index=0, key="afc_ind_x")
    with col2_ui:
        axis_y = st.selectbox("Axe Y:", [f"F{i}" for i in range(n_dims)], index=min(1, n_dims-1), key="afc_ind_y")

    if axis_y not in row_plot.columns:
        axis_y = axis_x

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=row_plot[axis_x], y=row_plot[axis_y] if axis_y in row_plot.columns else [0]*len(row_plot),
        mode='markers+text', text=row_plot.index.astype(str), textposition='top center',
        name=f'{var1} (lignes)', marker=dict(size=12, color='#667eea', symbol='circle')
    ))
    fig.add_trace(go.Scatter(
        x=col_plot[axis_x], y=col_plot[axis_y] if axis_y in col_plot.columns else [0]*len(col_plot),
        mode='markers+text', text=col_plot.index.astype(str), textposition='top center',
        name=f'{var2} (colonnes)', marker=dict(size=12, color='#EF4444', symbol='diamond')
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.add_vline(x=0, line_dash="dash", line_color="gray")
    fig.update_layout(title=f"Biplot AFC : {var1} × {var2}", xaxis_title=axis_x, yaxis_title=axis_y, height=600)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    <div class="info-box">
    <strong>Lecture AFC :</strong> Les modalités proches dans le plan sont fréquemment associées.
    Les modalités éloignées de l'origine sont les plus discriminantes.
    ● lignes (variable 1), ◆ colonnes (variable 2).
    </div>
    """, unsafe_allow_html=True)


# ==================== AFFICHAGE: VARIABLES ====================

def display_variables(model, col_coords, n_components, analysis_name, df_analysis):

    if analysis_name == "ACM":
        st.markdown("#### Projection des Modalités")
    elif analysis_name == "AFDM":
        st.markdown("#### Coordonnées des Variables (r² numériques / η² catégorielles)")
    else:
        st.markdown("#### Cercle des Corrélations" if analysis_name == "ACP" else "#### Projection des Variables")

    if col_coords is None or col_coords.empty:
        st.warning("Coordonnées des variables non disponibles pour ce modèle.")
        return

    n_dims = min(n_components, col_coords.shape[1])
    rename_dict = {old: f"F{i}" for i, old in enumerate(col_coords.columns[:n_dims])}
    col_coords_plot = col_coords.rename(columns=rename_dict)

    if 'F0' not in col_coords_plot.columns or 'F1' not in col_coords_plot.columns:
        st.warning("Pas assez de dimensions pour la projection des variables.")
        st.dataframe(col_coords_plot.round(4), use_container_width=True)
        return

    fig = px.scatter(
        col_coords_plot, x='F0', y='F1', text=col_coords_plot.index,
        title=("Cercle des Corrélations (F0-F1)" if analysis_name == "ACP"
               else f"Projection {'des Modalités' if analysis_name == 'ACM' else 'des Variables'} (F0-F1)")
    )
    fig.update_traces(textposition='top center', marker=dict(size=10, color='#667eea'))

    if analysis_name == "ACP":
        theta = np.linspace(0, 2*np.pi, 100)
        fig.add_trace(go.Scatter(x=np.cos(theta), y=np.sin(theta), mode='lines',
                                line=dict(color='gray', dash='dash'), showlegend=False, name='Cercle unité'))
        for idx in col_coords_plot.index:
            fig.add_trace(go.Scatter(
                x=[0, col_coords_plot.loc[idx, 'F0']], y=[0, col_coords_plot.loc[idx, 'F1']],
                mode='lines', line=dict(color='#667eea', width=1.5), showlegend=False
            ))
        fig.update_layout(xaxis=dict(range=[-1.1, 1.1]), yaxis=dict(range=[-1.1, 1.1], scaleanchor='x'))

    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.add_vline(x=0, line_dash="dash", line_color="gray")
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)

    if analysis_name == "AFDM":
        st.markdown("""
        <div class="info-box">
        <strong>Lecture AFDM :</strong> Les coordonnées des variables représentent :<br>
        • <strong>Variables numériques :</strong> r² (corrélation au carré) avec chaque axe.<br>
        • <strong>Variables catégorielles :</strong> η² (rapport de corrélation) — pouvoir discriminant sur l'axe.
        </div>
        """, unsafe_allow_html=True)

    if analysis_name == "ACM":
        st.markdown("""
        <div class="info-box">
        <strong>Lecture ACM :</strong> Chaque point représente une <strong>modalité</strong> (pas une variable).
        Les modalités proches sont souvent choisies ensemble par les mêmes individus.
        Les modalités éloignées de l'origine sont les plus discriminantes.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("#### Contributions aux Axes")
    contributions = safe_contributions(model, df_analysis)

    if contributions is not None:
        contributions = contributions.copy()
        contributions.columns = [f"F{i}" for i in range(contributions.shape[1])]
        st.dataframe(contributions.round(4), use_container_width=True)
    else:
        st.info("Contributions non disponibles. Affichage des coordonnées à la place.")
        st.dataframe(col_coords_plot.round(4), use_container_width=True)


def display_variables_afc(model, row_coords, col_coords, var1, var2, contingency_table):
    st.markdown("#### Tableau de Contingence")
    st.dataframe(contingency_table, use_container_width=True)

    chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
    with st.expander("📊 Effectifs théoriques sous H₀ (indépendance)"):
        expected_df = pd.DataFrame(expected, index=contingency_table.index, columns=contingency_table.columns)
        st.dataframe(expected_df.round(2), use_container_width=True)

    st.markdown("#### Profils Lignes et Colonnes")
    col1, col2 = st.columns(2)
    with col1:
        row_profiles = contingency_table.div(contingency_table.sum(axis=1), axis=0)
        st.markdown(f"**Profils lignes ({var1}) :**")
        st.dataframe(row_profiles.round(3), use_container_width=True)
    with col2:
        col_profiles = contingency_table.div(contingency_table.sum(axis=0), axis=1)
        st.markdown(f"**Profils colonnes ({var2}) :**")
        st.dataframe(col_profiles.round(3), use_container_width=True)

    st.markdown("#### Profil moyen (masses)")
    col1, col2 = st.columns(2)
    with col1:
        row_masses = contingency_table.sum(axis=1) / contingency_table.sum().sum()
        st.markdown(f"**Masses lignes ({var1}) :**")
        st.dataframe(row_masses.round(4).rename("masse"), use_container_width=True)
    with col2:
        col_masses = contingency_table.sum(axis=0) / contingency_table.sum().sum()
        st.markdown(f"**Masses colonnes ({var2}) :**")
        st.dataframe(col_masses.round(4).rename("masse"), use_container_width=True)


# ==================== AFFICHAGE: INTERPRÉTATION ====================

def display_interpretation(model, col_coords, n_components, analysis_name, df_analysis):
    st.markdown("#### Interprétation des Axes Factoriels")

    if analysis_name == "ACP":
        st.markdown("""
        **Principe :** Chaque axe est une combinaison linéaire des variables originales.
        Les variables les plus corrélées (positivement ou négativement) à un axe le définissent.

        **Axe F0 :** Variables à forte corrélation positive à droite, négative à gauche.
        **Axe F1 :** Variables à forte corrélation positive en haut, négative en bas.
        """)
    elif analysis_name == "ACM":
        st.markdown("""
        **Principe :** Chaque axe oppose des groupes de modalités. Les modalités rares
        (peu fréquentes) ont tendance à contribuer davantage aux axes.

        **Axe F0 :** Opposition entre deux profils de réponses.
        **Axe F1 :** Second contraste entre profils de réponses.
        """)
    elif analysis_name == "AFC":
        st.markdown("""
        **Principe :** Chaque axe explique une part de l'écart à l'indépendance entre
        les deux variables. Les modalités proches dans le biplot sont associées plus
        fréquemment que sous l'hypothèse d'indépendance.
        """)
    else:
        st.markdown("""
        **Principe :** L'AFDM combine ACP (variables numériques) et ACM (variables catégorielles).
        Chaque axe résume à la fois des oppositions numériques et des associations entre modalités.

        **Coordonnées :** r² pour les variables numériques, η² pour les catégorielles.
        """)

    contributions = safe_contributions(model, df_analysis)
    if contributions is not None:
        st.markdown("##### Top Variables/Modalités Contributrices par Axe")
        contributions = contributions.copy()
        contributions.columns = [f"F{i}" for i in range(contributions.shape[1])]

        for i in range(min(3, contributions.shape[1])):
            with st.expander(f"📊 Axe F{i}"):
                top_contrib = contributions[f'F{i}'].sort_values(ascending=False).head(10)
                fig = px.bar(x=top_contrib.values, y=top_contrib.index, orientation='h',
                            title=f"Top 10 Contributions à F{i}", color=top_contrib.values,
                            color_continuous_scale='Viridis')
                fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig, use_container_width=True)
    elif col_coords is not None and not col_coords.empty:
        st.markdown("##### Coordonnées des Variables (proxy des contributions)")
        col_coords_disp = col_coords.copy()
        col_coords_disp.columns = [f"F{i}" for i in range(col_coords_disp.shape[1])]

        for i in range(min(3, col_coords_disp.shape[1])):
            with st.expander(f"📊 Axe F{i}"):
                top_abs = col_coords_disp[f'F{i}'].abs().sort_values(ascending=False).head(10)
                top_vals = col_coords_disp.loc[top_abs.index, f'F{i}']
                fig = px.bar(x=top_vals.values, y=top_vals.index, orientation='h',
                            title=f"Top 10 Coordonnées (F{i})", color=top_vals.values,
                            color_continuous_scale='RdBu_r')
                fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Contributions non disponibles pour ce type d'analyse.")


# ==================== PAGE: CLUSTERING (COMMENTÉE) ====================

# def page_clustering(df):
#     st.markdown("## 🎯 Clustering des Employés")
#     st.markdown('<div class="info-box"><strong>Objectif :</strong> Identifier des groupes homogènes d\'employés avec K-Means et CAH.</div>', unsafe_allow_html=True)
#
#     df_processed = preprocess_for_analysis(df)
#     df_encoded, label_encoders = encode_categorical(df_processed)
#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(df_encoded)
#
#     tab1, tab2, tab3 = st.tabs(["📊 K-Means", "🌳 CAH", "📈 Profilage"])
#
#     with tab1:
#         st.markdown("### K-Means Clustering")
#         col1, col2 = st.columns(2)
#         with col1:
#             st.markdown("#### Méthode du Coude")
#             inertias, silhouettes = [], []
#             K_range = range(2, 11)
#             for k in K_range:
#                 km = KMeans(n_clusters=k, random_state=42, n_init=10)
#                 km.fit(X_scaled)
#                 inertias.append(km.inertia_)
#                 silhouettes.append(silhouette_score(X_scaled, km.labels_))
#
#             fig = make_subplots(specs=[[{"secondary_y": True}]])
#             fig.add_trace(go.Scatter(x=list(K_range), y=inertias, name="Inertie",
#                                     mode='lines+markers', marker_color='#667eea'), secondary_y=False)
#             fig.add_trace(go.Scatter(x=list(K_range), y=silhouettes, name="Silhouette",
#                                     mode='lines+markers', marker_color='#EF4444'), secondary_y=True)
#             fig.update_layout(title="Coude & Silhouette")
#             fig.update_xaxes(title_text="K")
#             fig.update_yaxes(title_text="Inertie", secondary_y=False)
#             fig.update_yaxes(title_text="Silhouette", secondary_y=True)
#             st.plotly_chart(fig, use_container_width=True)
#         with col2:
#             n_clusters = st.slider("Nombre de clusters K:", 2, 10, 4)
#
#         if st.button("🚀 Appliquer K-Means", type="primary"):
#             with st.spinner("Clustering en cours..."):
#                 kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
#                 clusters = kmeans.fit_predict(X_scaled)
#                 df_clustered = df_processed.copy()
#                 df_clustered['Cluster'] = clusters
#
#                 pca_2d = PCA(n_components=2)
#                 X_pca = pca_2d.fit_transform(X_scaled)
#                 df_viz = pd.DataFrame({'PC1': X_pca[:, 0], 'PC2': X_pca[:, 1],
#                                       'Cluster': clusters.astype(str), 'Attrition': df_processed['Attrition']})
#
#                 col1, col2 = st.columns(2)
#                 with col1:
#                     fig = px.scatter(df_viz, x='PC1', y='PC2', color='Cluster',
#                                     title="Clusters (PCA)", color_discrete_sequence=px.colors.qualitative.Set1)
#                     st.plotly_chart(fig, use_container_width=True)
#                 with col2:
#                     fig = px.scatter(df_viz, x='PC1', y='PC2', color='Attrition',
#                                     title="Attrition (PCA)", color_discrete_map={'No': '#10B981', 'Yes': '#EF4444'})
#                     st.plotly_chart(fig, use_container_width=True)
#
#                 col1, col2, col3 = st.columns(3)
#                 col1.metric("Score Silhouette", f"{silhouette_score(X_scaled, clusters):.3f}")
#                 col2.metric("Inertie", f"{kmeans.inertia_:.2f}")
#                 col3.metric("Clusters", n_clusters)
#
#                 st.session_state['clusters'] = clusters
#                 st.session_state['df_clustered'] = df_clustered
#                 st.success("✅ Clustering terminé! Consultez l'onglet Profilage.")
#
#     with tab2:
#         st.markdown("### Classification Ascendante Hiérarchique (CAH)")
#         sample_size = st.slider("Taille de l'échantillon:", 100, min(1000, len(df)), 500)
#         linkage_method = st.selectbox("Méthode de liaison:", ['ward', 'complete', 'average', 'single'])
#
#         if st.button("🌳 Générer le Dendrogramme"):
#             with st.spinner("Calcul du dendrogramme..."):
#                 indices = np.random.choice(len(X_scaled), sample_size, replace=False)
#                 X_sample = X_scaled[indices]
#                 Z = linkage(X_sample, method=linkage_method)
#                 fig, ax = plt.subplots(figsize=(12, 6))
#                 dendrogram(Z, truncate_mode='level', p=5, ax=ax)
#                 ax.set_title("Dendrogramme (CAH)")
#                 ax.set_xlabel("Individus")
#                 ax.set_ylabel("Distance")
#                 st.pyplot(fig)
#
#     with tab3:
#         st.markdown("### Profilage des Clusters")
#         if 'df_clustered' in st.session_state:
#             df_clustered = st.session_state['df_clustered']
#             attrition_by_cluster = df_clustered.groupby('Cluster')['Attrition'].value_counts(normalize=True).unstack()
#             fig = px.bar(attrition_by_cluster, barmode='group', title="Taux d'Attrition par Cluster")
#             st.plotly_chart(fig, use_container_width=True)
#
#             quant_vars, _ = get_variable_types(df_clustered)
#             quant_vars = [v for v in quant_vars if v != 'Cluster']
#             cluster_means = df_clustered.groupby('Cluster')[quant_vars].mean()
#             cluster_means_norm = (cluster_means - cluster_means.min()) / (cluster_means.max() - cluster_means.min() + 1e-9)
#
#             default_radar = [v for v in ['Age', 'MonthlyIncome', 'YearsAtCompany', 'JobSatisfaction', 'WorkLifeBalance'] if v in quant_vars]
#             selected_vars = st.multiselect("Variables pour le profil radar:", quant_vars, default=default_radar)
#
#             if selected_vars:
#                 fig = go.Figure()
#                 for cluster in cluster_means_norm.index:
#                     fig.add_trace(go.Scatterpolar(
#                         r=cluster_means_norm.loc[cluster, selected_vars].values,
#                         theta=selected_vars, fill='toself', name=f'Cluster {cluster}'
#                     ))
#                 fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), title="Profil des Clusters")
#                 st.plotly_chart(fig, use_container_width=True)
#
#             st.dataframe(cluster_means.round(2), use_container_width=True)
#         else:
#             st.warning("Veuillez d'abord effectuer le clustering K-Means.")


# ==================== PAGE: CLASSIFICATION ====================

def page_classification(df):
    st.markdown("## 🤖 Prédiction de l'Attrition")

    df_processed = preprocess_for_analysis(df)
    df_encoded, label_encoders = encode_categorical(df_processed)

    X = df_encoded.drop('Attrition', axis=1)
    y = df_encoded['Attrition']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    tab1, tab2, tab3 = st.tabs(["🎯 Entraînement", "📊 Évaluation", "🔍 Interprétabilité"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            model_choice = st.selectbox("Algorithme:", ["Random Forest", "Gradient Boosting", "Logistic Regression"])
            use_smote = st.checkbox("Utiliser SMOTE", value=True)
        with col2:
            n_estimators = st.slider("Estimateurs / Itérations:", 50, 300, 100)
            if model_choice == "Random Forest":
                max_depth = st.slider("Profondeur max:", 3, 20, 10)
            elif model_choice == "Gradient Boosting":
                learning_rate = st.slider("Learning rate:", 0.01, 0.3, 0.1)

        if st.button("🚀 Entraîner le Modèle", type="primary"):
            with st.spinner("Entraînement en cours..."):
                if use_smote:
                    smote = SMOTE(random_state=42)
                    X_bal, y_bal = smote.fit_resample(X_train_scaled, y_train)
                    st.info(f"SMOTE : {len(y_train)} → {len(y_bal)} échantillons")
                else:
                    X_bal, y_bal = X_train_scaled, y_train

                if model_choice == "Random Forest":
                    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                                  random_state=42, class_weight='balanced')
                elif model_choice == "Gradient Boosting":
                    model = GradientBoostingClassifier(n_estimators=n_estimators,
                                                      learning_rate=learning_rate, random_state=42)
                else:
                    model = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)

                model.fit(X_bal, y_bal)
                y_pred = model.predict(X_test_scaled)
                y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
                cv_scores = cross_val_score(model, X_bal, y_bal, cv=5, scoring='f1')

                st.session_state['model'] = model
                st.session_state['y_test'] = y_test
                st.session_state['y_pred'] = y_pred
                st.session_state['y_pred_proba'] = y_pred_proba
                st.session_state['feature_names'] = X.columns.tolist()
                st.session_state['cv_scores'] = cv_scores
                st.success(f"✅ F1-score CV : {cv_scores.mean():.3f} (±{cv_scores.std():.3f})")

    with tab2:
        if 'model' in st.session_state:
            y_test = st.session_state['y_test']
            y_pred = st.session_state['y_pred']
            y_pred_proba = st.session_state['y_pred_proba']

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.3f}")
            col2.metric("Precision", f"{precision_score(y_test, y_pred):.3f}")
            col3.metric("Recall", f"{recall_score(y_test, y_pred):.3f}")
            col4.metric("F1-Score", f"{f1_score(y_test, y_pred):.3f}")

            col1, col2 = st.columns(2)
            with col1:
                cm = confusion_matrix(y_test, y_pred)
                fig = px.imshow(cm, labels=dict(x="Prédit", y="Réel"),
                               x=['Reste', 'Part'], y=['Reste', 'Part'],
                               color_continuous_scale='Blues', text_auto=True, title="Matrice de Confusion")
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
                roc_auc = auc(fpr, tpr)
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC (AUC={roc_auc:.3f})', line=dict(color='#667eea')))
                fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Aléatoire', line=dict(color='gray', dash='dash')))
                fig.update_layout(title="Courbe ROC", xaxis_title="FPR", yaxis_title="TPR")
                st.plotly_chart(fig, use_container_width=True)

            report = classification_report(y_test, y_pred, target_names=['Reste', 'Part'], output_dict=True)
            st.dataframe(pd.DataFrame(report).T.round(3), use_container_width=True)
        else:
            st.warning("Veuillez d'abord entraîner un modèle.")

    with tab3:
        if 'model' in st.session_state:
            model = st.session_state['model']
            feature_names = st.session_state['feature_names']

            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
            elif hasattr(model, 'coef_'):
                importances = np.abs(model.coef_[0])
            else:
                importances = None

            if importances is not None:
                importance_df = pd.DataFrame({'Variable': feature_names, 'Importance': importances})
                importance_df = importance_df.sort_values('Importance', ascending=False)
                top_features = importance_df.head(15)
                fig = px.bar(top_features, x='Importance', y='Variable', orientation='h',
                            title="Top 15 Variables Importantes", color='Importance', color_continuous_scale='Viridis')
                fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig, use_container_width=True)

                top_3 = importance_df.head(3)['Variable'].tolist()
                st.markdown(f"""
                <div class="success-box">
                <strong>Variables clés :</strong>
                <ol><li>{top_3[0]}</li><li>{top_3[1]}</li><li>{top_3[2]}</li></ol>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.warning("Veuillez d'abord entraîner un modèle.")


# ==================== PAGE: CONCLUSION ====================

def page_conclusion():
    st.markdown("## 📝 Conclusion et Recommandations")

    st.markdown("""
    <div class="info-box">
    <strong>Contexte :</strong> Analyse du dataset IBM HR Analytics — <strong>1 470 employés</strong>,
    <strong>35 variables</strong>, taux d'attrition de <strong>16,1%</strong>.
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("👥 Employés", "1 470")
    col2.metric("🚪 Départs", "237", "16.1%")
    col3.metric("📋 Variables", "35")
    col4.metric("💰 Salaire Médian", "4 919$")

    st.markdown("---")
    st.markdown("### 🔍 Facteurs Clés d'Attrition")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""<div class="warning-box"><strong>⏰ Heures Supplémentaires</strong><br>
        30.5% d'attrition avec heures sup vs 10.4% sans.</div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""<div class="warning-box"><strong>💵 Revenu Mensuel</strong><br>
        Partis : 4 787$ en moyenne vs 6 833$ pour les restants.</div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""<div class="warning-box"><strong>📅 Ancienneté</strong><br>
        Attrition plus forte avec moins de 2 ans d'ancienneté.</div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 💡 Recommandations")
    st.markdown("""
    <div class="success-box">
    <ol>
    <li><strong>Réduire les heures supplémentaires</strong> — facteur #1 d'attrition</li>
    <li><strong>Révision salariale</strong> pour les employés sous 5 000$/mois</li>
    <li><strong>Programme d'intégration renforcé</strong> les 2 premières années</li>
    <li><strong>Plan de carrière clair</strong> pour les 25–35 ans</li>
    <li><strong>Enquêtes de satisfaction</strong> régulières et actions correctives</li>
    </ol>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🔬 Justification du Choix du Modèle d'Analyse")
    st.markdown("""
    <div class="info-box">
    <strong>Modèle principal : AFDM (Analyse Factorielle des Données Mixtes)</strong><br><br>
    Le choix de l'AFDM comme méthode principale se justifie par la <strong>nature mixte</strong> du dataset :
    <ul>
    <li><strong>Variables quantitatives</strong> (Age, MonthlyIncome, YearsAtCompany…) → traitées comme en ACP (centrée-réduite)</li>
    <li><strong>Variables qualitatives</strong> (Department, JobRole, OverTime…) → traitées comme en ACM (codage disjonctif)</li>
    </ul>
    L'AFDM équilibre les contributions des deux types de variables grâce à une pondération spécifique, 
    évitant que les variables numériques (plus nombreuses) ne dominent l'analyse.<br><br>
    <strong>Modèles complémentaires proposés :</strong>
    <ul>
    <li><strong>ACP centrée-réduite :</strong> Pour analyser finement les corrélations entre variables numériques 
    (échelles différentes → standardisation nécessaire).</li>
    <li><strong>ACM avec correction de Benzécri :</strong> Pour explorer les associations entre modalités qualitatives 
    (correction nécessaire car les valeurs propres brutes de l'ACM sont sous-estimées).</li>
    <li><strong>AFC :</strong> Pour étudier des associations bivariées spécifiques (ex. OverTime × Attrition).</li>
    </ul>
    <strong>Outil utilisé :</strong> Bibliothèque <code>prince</code> (Python), spécialisée en analyses factorielles.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### ✍️ Synthèse Individuelle — Difficultés Rencontrées")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="stat-card">
        <h4>Mohammed Tbahriti</h4>
        <strong>Contributions :</strong> Architecture de l'application Streamlit, implémentation de l'AFDM et de l'ACP, 
        pipeline de machine learning (SMOTE, Random Forest, validation croisée).<br><br>
        <strong>Difficultés :</strong>
        <ul>
        <li>La bibliothèque <code>prince</code> a changé d'API entre versions, ce qui a nécessité des fonctions 
        d'adaptation (<code>safe_column_coordinates</code>, etc.) pour garantir la compatibilité.</li>
        <li>Le déséquilibre des classes (16% d'attrition) faussait fortement les métriques de classification ; 
        l'intégration de SMOTE et l'utilisation du F1-score ont résolu ce problème.</li>
        <li>La gestion des variables ordinales (Education, JobSatisfaction) qui peuvent être traitées 
        comme quantitatives ou qualitatives selon le contexte d'analyse.</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="stat-card">
        <h4>Youcef Moulai</h4>
        <strong>Contributions :</strong> Exploration des données (EDA), détection des outliers, 
        implémentation de l'ACM et de l'AFC, analyse des corrélations.<br><br>
        <strong>Difficultés :</strong>
        <ul>
        <li>En ACM, les valeurs propres brutes sont très faibles (structurellement sous-estimées), 
        ce qui rend l'interprétation de l'inertie expliquée contre-intuitive. Les corrections 
        de Benzécri et Greenacre ont été essentielles.</li>
        <li>Le choix des variables qualitatives pertinentes pour l'ACM : certaines variables 
        ordinales (1-5) donnent de meilleurs résultats en ACP qu'en ACM.</li>
        <li>La lecture du biplot AFC nécessite de bien comprendre les profils-lignes 
        et profils-colonnes pour une interprétation correcte.</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="stat-card">
        <h4>Yahia Ouahmed Yanis</h4>
        <strong>Contributions :</strong> Clustering (K-Means, CAH), profilage des clusters, 
        validation par silhouette score, visualisation des résultats.<br><br>
        <strong>Difficultés :</strong>
        <ul>
        <li>Le choix du nombre optimal de clusters : la méthode du coude n'est pas toujours claire, 
        le silhouette score et l'interprétabilité métier ont guidé le choix final.</li>
        <li>L'encodage des variables catégorielles pour le clustering (LabelEncoder vs OneHotEncoder) 
        impacte significativement les résultats.</li>
        <li>Le dendrogramme CAH sur 1 470 individus est illisible ; 
        un sous-échantillonnage a été nécessaire pour la visualisation.</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
    <strong>Questions ouvertes :</strong>
    <ul>
    <li>L'AFDM donne-t-elle toujours de meilleurs résultats que l'ACP + ACM séparées ?</li>
    <li>Comment interpréter les axes factoriels quand les contributions sont réparties uniformément ?</li>
    <li>La projection des points supplémentaires est-elle fiable quand le taux d'attrition est très déséquilibré ?</li>
    <li>Quelle est la robustesse des clusters identifiés face à des perturbations des données ?</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)


# ==================== MAIN ====================

def main():
    st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/5/51/IBM_logo.svg", width=100)
    st.sidebar.title("📊 Navigation")

    page = st.sidebar.radio(
        "Aller à:",
        ["🏠 Accueil", "📊 Exploration (EDA)", "🔬 Analyse Factorielle",
         # "🎯 Clustering",
         "🤖 Classification", "📝 Conclusion"]
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 👥 Équipe")
    st.sidebar.markdown("- Mohammed Tbahriti\n- Youcef Moulai\n- Yahia Ouahmed Yanis")
    st.sidebar.markdown("---")
    st.sidebar.markdown("*Master 2 IA — URCA*")
    st.sidebar.markdown("*INFO0902 — Analyse des Données*")

    df = load_data()

    if df is not None:
        if page == "🏠 Accueil":
            page_accueil()
        elif page == "📊 Exploration (EDA)":
            page_exploration(df)
        elif page == "🔬 Analyse Factorielle":
            page_afdm(df)
        # elif page == "🎯 Clustering":
        #     page_clustering(df)
        elif page == "🤖 Classification":
            page_classification(df)
        elif page == "📝 Conclusion":
            page_conclusion()
    else:
        st.error("Dataset non trouvé. Placez le fichier CSV dans le dossier du projet.")


if __name__ == "__main__":
    main()