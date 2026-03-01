# 🎯 IBM HR Analytics - Analyse de l'Attrition des Employés

## Projet INFO0902 - Analyse des Données
### Master 2 Intelligence Artificielle & Data Science
### Université de Reims Champagne-Ardenne - Année 2025-2026

---

## 📋 Description du Projet

Ce projet vise à analyser les facteurs influençant l'attrition des employés dans une entreprise fictive d'IBM. Nous utilisons une approche complète combinant :

- **Analyse Factorielle des Données Mixtes (AFDM)** pour réduire la dimensionnalité et identifier les axes structurants
- **Clustering (K-Means & CAH)** pour segmenter les employés en groupes homogènes
- **Machine Learning (Classification)** pour prédire le risque de départ

## 🎯 Problématique

**Question principale :** Quels sont les facteurs déterminants qui influencent la décision d'un employé de quitter l'entreprise, et comment peut-on prédire et prévenir l'attrition ?

## 📁 Structure du Projet

```
hr_analytics_project/
│
├── app.py                  # Application Streamlit principale
├── requirements.txt        # Dépendances Python
├── README.md              # Ce fichier
├── data/
│   └── WA_Fn-UseC_-HR-Employee-Attrition.csv  # Dataset (à télécharger)
└── rapport/
    └── rapport_projet.pdf  # Rapport final
```

## 🛠️ Installation

### Prérequis
- Python 3.9+
- pip

### Étapes d'installation

```bash
# 1. Cloner ou télécharger le projet
cd hr_analytics_project

# 2. Créer un environnement virtuel (recommandé)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate     # Windows

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Télécharger le dataset depuis Kaggle
# https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset
# Placer le fichier CSV dans le dossier data/

# 5. Lancer l'application
streamlit run app.py
```

## 📊 Dataset

**IBM HR Analytics Employee Attrition & Performance**
- Source : [Kaggle](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)
- 1470 observations
- 35 variables

### Variables principales

| Variable | Type | Description |
|----------|------|-------------|
| Attrition | Qualitative | Variable cible (Yes/No) |
| Age | Quantitative | Âge de l'employé |
| MonthlyIncome | Quantitative | Salaire mensuel |
| YearsAtCompany | Quantitative | Années d'ancienneté |
| OverTime | Qualitative | Heures supplémentaires (Yes/No) |
| Department | Qualitative | Département |
| JobRole | Qualitative | Poste occupé |
| JobSatisfaction | Quantitative | Satisfaction au travail (1-4) |
| WorkLifeBalance | Quantitative | Équilibre vie pro/perso (1-4) |

## 🔬 Méthodologie

### 1. Exploration des Données (EDA)
- Statistiques descriptives
- Analyse des distributions
- Détection des outliers
- Matrice de corrélation

### 2. Analyse Factorielle (4 Méthodes au Choix)

L'application propose **4 méthodes d'analyse factorielle** que vous pouvez sélectionner via un menu déroulant :

| Méthode | Acronyme | Type de Données | Usage |
|---------|----------|-----------------|-------|
| **ACP** | Analyse en Composantes Principales | Quantitatives uniquement | Réduction de dimensionnalité pour variables numériques |
| **ACM** | Analyse des Correspondances Multiples | Qualitatives uniquement | Analyse des associations entre modalités |
| **AFDM** | Analyse Factorielle des Données Mixtes | Mixtes (quanti + quali) | Combine ACP et ACM pour données hétérogènes |
| **AFC** | Analyse Factorielle des Correspondances | 2 variables qualitatives | Analyse d'un tableau de contingence |

**Justification pour ce dataset :**
- Le dataset IBM HR contient des variables quantitatives (Age, Income...) ET qualitatives (Department, MaritalStatus...)
- L'**AFDM** est recommandée pour ce type de données mixtes
- Mais l'utilisateur peut aussi tester l'**ACP** sur les variables numériques seules ou l'**ACM** sur les catégorielles

### 3. Clustering
- **K-Means** avec méthode du coude et score silhouette
- **Classification Ascendante Hiérarchique (CAH)**
- Profilage des clusters identifiés

### 4. Classification (Prédiction)
- Modèles testés : Random Forest, Gradient Boosting, Logistic Regression
- Gestion du déséquilibre des classes avec SMOTE
- Validation croisée stratifiée
- Analyse de l'importance des variables

## 📈 Résultats Attendus

1. **AFDM** : Identification des dimensions principales structurant les profils d'employés
2. **Clustering** : 3-5 segments d'employés avec des profils distincts
3. **Classification** : Modèle prédictif avec F1-score > 0.70

## 💻 Technologies Utilisées

- **Python 3.9+**
- **Streamlit** - Interface web interactive
- **Pandas & NumPy** - Manipulation des données
- **Scikit-learn** - Machine Learning
- **Prince** - Analyse factorielle (AFDM)
- **Plotly & Seaborn** - Visualisations
- **SMOTE (imbalanced-learn)** - Rééquilibrage des classes

## 👥 Équipe

| Nom | Prénom | Contribution |
|-----|--------|--------------|
| ... | ... | EDA, Prétraitement |
| ... | ... | AFDM, Interprétation |
| ... | ... | Clustering, Classification |

## 📝 Rapport

Le rapport final comprend :
1. Introduction et contexte
2. Description du dataset et problématique
3. Méthodologie détaillée
4. Résultats et interprétations
5. Conclusion et recommandations
6. Synthèse individuelle de chaque membre

## 📚 Références

- Escofier, B., & Pagès, J. (2008). *Analyses factorielles simples et multiples*. Dunod.
- Lebart, L., Piron, M., & Morineau, A. (2006). *Statistique exploratoire multidimensionnelle*. Dunod.
- Documentation Prince : https://github.com/MaxHalford/prince
- Documentation Scikit-learn : https://scikit-learn.org/

---

*Projet réalisé dans le cadre du cours INFO0902 - Analyse des Données*
*Master 2 IA & Data Science - Université de Reims Champagne-Ardenne*
