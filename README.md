# 🎯 IBM HR Analytics - Analyse de l'Attrition des Employés

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/Streamlit-1.28+-red.svg" alt="Streamlit">
  <img src="https://img.shields.io/badge/Scikit--learn-1.3+-orange.svg" alt="Sklearn">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
</p>

## 📋 Informations du Projet

| | |
|---|---|
| **Cours** | INFO0902 - Analyse des Données |
| **Formation** | Master 2 Intelligence Artificielle & Data Science |
| **Université** | Université de Reims Champagne-Ardenne (URCA) |
| **Année** | 2025-2026 |

---

## 📖 Table des Matières

1. [Description du Projet](#-description-du-projet)
2. [Problématique](#-problématique)
3. [Dataset](#-dataset)
4. [Méthodologie](#-méthodologie)
5. [Fonctionnalités de l'Application](#-fonctionnalités-de-lapplication)
6. [Installation](#-installation)
7. [Utilisation](#-utilisation)
8. [Structure du Projet](#-structure-du-projet)
9. [Résultats Attendus](#-résultats-attendus)
10. [Technologies Utilisées](#-technologies-utilisées)
11. [Équipe](#-équipe)
12. [Références](#-références)

---

## 🎯 Description du Projet

Ce projet vise à analyser les facteurs influençant **l'attrition des employés** (départ volontaire) dans une entreprise fictive d'IBM. L'objectif est triple :

1. **Comprendre** les variables qui impactent la décision de quitter l'entreprise
2. **Segmenter** les employés en groupes homogènes pour identifier les profils à risque
3. **Prédire** le risque d'attrition pour permettre des actions préventives

L'application web interactive développée avec **Streamlit** permet d'explorer les données, d'effectuer différentes analyses factorielles, de réaliser du clustering et de construire des modèles prédictifs.

---

## ❓ Problématique

### Question Principale
> **Quels sont les facteurs déterminants qui influencent la décision d'un employé de quitter l'entreprise, et comment peut-on prédire et prévenir l'attrition ?**

### Questions Secondaires
- Existe-t-il des profils types d'employés à risque de départ ?
- Quelles variables ont le plus d'impact sur la satisfaction et la rétention ?
- Peut-on identifier des clusters d'employés ayant des caractéristiques similaires ?
- Quel modèle de machine learning est le plus performant pour prédire l'attrition ?

### Enjeux Métier
- **Coût du turnover** : Recrutement, formation, perte de productivité
- **Rétention des talents** : Identifier et retenir les employés clés
- **Amélioration RH** : Adapter les politiques de ressources humaines

---

## 📁 Dataset

### Source
**IBM HR Analytics Employee Attrition & Performance**
- 🔗 [Kaggle](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)
- Dataset fictif créé par les data scientists d'IBM

### Caractéristiques
| Caractéristique | Valeur |
|-----------------|--------|
| Nombre d'observations | 1 470 |
| Nombre de variables | 35 |
| Variables quantitatives | 26 |
| Variables qualitatives | 9 |
| Variable cible | Attrition (Yes/No) |
| Valeurs manquantes | 0 |

### Variables Principales

#### Variables Quantitatives
| Variable | Description | Plage |
|----------|-------------|-------|
| `Age` | Âge de l'employé | 18-60 |
| `MonthlyIncome` | Salaire mensuel | 1 000 - 20 000 |
| `YearsAtCompany` | Années d'ancienneté | 0-40 |
| `TotalWorkingYears` | Expérience totale | 0-40 |
| `DistanceFromHome` | Distance domicile-travail | 1-29 |
| `JobSatisfaction` | Satisfaction au travail | 1-4 |
| `WorkLifeBalance` | Équilibre vie pro/perso | 1-4 |
| `EnvironmentSatisfaction` | Satisfaction environnement | 1-4 |
| `PercentSalaryHike` | Augmentation salariale (%) | 11-25 |

#### Variables Qualitatives
| Variable | Description | Modalités |
|----------|-------------|-----------|
| `Attrition` | **Variable cible** | Yes, No |
| `Department` | Département | Sales, R&D, HR |
| `JobRole` | Poste | 9 modalités |
| `Gender` | Genre | Male, Female |
| `MaritalStatus` | Statut marital | Single, Married, Divorced |
| `OverTime` | Heures supplémentaires | Yes, No |
| `BusinessTravel` | Fréquence des déplacements | 3 modalités |
| `EducationField` | Domaine d'études | 6 modalités |

### Déséquilibre des Classes
```
Attrition = No  : 84% (1 233 employés)
Attrition = Yes : 16% (237 employés)
```
⚠️ Ce déséquilibre nécessite l'utilisation de techniques comme **SMOTE** pour la classification.

---

## 🔬 Méthodologie

### 1. Analyse Exploratoire des Données (EDA)

| Étape | Description |
|-------|-------------|
| Statistiques descriptives | Moyenne, médiane, écart-type, quartiles |
| Visualisation des distributions | Histogrammes, boxplots, diagrammes en barres |
| Analyse des corrélations | Matrice de corrélation, top corrélations avec Attrition |
| Détection des outliers | Méthode IQR (Interquartile Range) |

### 2. Analyse Factorielle (4 Méthodes au Choix)

L'application propose **4 méthodes d'analyse factorielle** sélectionnables via un menu :

| Méthode | Acronyme | Type de Données | Quand l'utiliser |
|---------|----------|-----------------|------------------|
| **ACP** | Analyse en Composantes Principales | Quantitatives uniquement | Variables numériques (Age, Income...) |
| **ACM** | Analyse des Correspondances Multiples | Qualitatives uniquement | Variables catégorielles (Department...) |
| **AFDM** | Analyse Factorielle des Données Mixtes | Mixtes (quanti + quali) | ✅ **Recommandé pour ce dataset** |
| **AFC** | Analyse Factorielle des Correspondances | 2 variables qualitatives | Tableau de contingence |

#### Justification du Choix de l'AFDM
Le dataset IBM HR contient des variables **quantitatives** (Age, MonthlyIncome, YearsAtCompany...) **ET qualitatives** (Department, MaritalStatus, OverTime...). L'**AFDM** est la méthode la plus appropriée car elle :
- Combine les principes de l'ACP (pour les variables quantitatives)
- Et de l'ACM (pour les variables qualitatives)
- Permet une analyse globale des données mixtes

#### Résultats de l'Analyse Factorielle
- **Valeurs propres** et pourcentage d'inertie expliquée
- **Éboulis des valeurs propres** (Scree plot)
- **Projection des individus** sur les plans factoriels
- **Projection des variables** et cercle des corrélations
- **Contributions** des variables aux axes

### 3. Clustering

| Méthode | Description |
|---------|-------------|
| **K-Means** | Partitionnement en K clusters |
| **Méthode du coude** | Choix optimal du nombre de clusters |
| **Score Silhouette** | Validation de la qualité des clusters |
| **CAH** | Classification Ascendante Hiérarchique avec dendrogramme |
| **Profilage** | Caractérisation de chaque cluster |

#### Critères de Choix du Nombre de Clusters
1. **Critère du coude** : Point d'inflexion de l'inertie intra-classe
2. **Score silhouette** : Maximiser la cohésion intra-cluster
3. **Interprétabilité** : Nombre de clusters permettant une lecture métier

### 4. Classification (Prédiction de l'Attrition)

| Algorithme | Type | Avantages |
|------------|------|-----------|
| **Random Forest** | Ensemble (bagging) | Robuste, feature importance |
| **Gradient Boosting** | Ensemble (boosting) | Performance élevée |
| **Logistic Regression** | Linéaire | Interprétable, baseline |

#### Gestion du Déséquilibre
- **SMOTE** (Synthetic Minority Over-sampling Technique)
- **Class weights** pour pénaliser les erreurs sur la classe minoritaire

#### Métriques d'Évaluation
| Métrique | Description |
|----------|-------------|
| Accuracy | Taux de bonnes prédictions |
| Precision | Parmi les prédits positifs, combien sont vrais |
| Recall | Parmi les vrais positifs, combien sont détectés |
| F1-Score | Moyenne harmonique precision/recall |
| AUC-ROC | Aire sous la courbe ROC |

---

## 🖥️ Fonctionnalités de l'Application

### Navigation

```
┌─────────────────────────────────────────────────────────────┐
│  🏠 Accueil                                                 │
│  📊 Exploration (EDA)                                       │
│  🔬 Analyse Factorielle  ←  ACP / ACM / AFDM / AFC         │
│  🎯 Clustering                                              │
│  🤖 Classification                                          │
│  📝 Conclusion                                              │
└─────────────────────────────────────────────────────────────┘
```

### 🏠 Page Accueil
- Présentation du projet et de la problématique
- Description de la méthodologie
- Aperçu des techniques utilisées

### 📊 Page Exploration (EDA)
- **Aperçu** : Shape, types, statistiques descriptives
- **Distributions** : Histogrammes, pie charts, bar plots
- **Corrélations** : Heatmap, top corrélations avec Attrition
- **Outliers** : Boxplots, comptage par variable

### 🔬 Page Analyse Factorielle
- **Sélection du modèle** : ACP, ACM, AFDM ou AFC
- **Configuration** : Nombre de composantes, variables
- **Résultats** :
  - Tableau des valeurs propres
  - Éboulis (Scree plot)
  - Projection des individus (colorés par Attrition)
  - Projection des variables
  - Contributions aux axes

### 🎯 Page Clustering
- **K-Means** : Méthode du coude, silhouette
- **CAH** : Dendrogramme interactif
- **Profilage** : Radar chart, statistiques par cluster

### 🤖 Page Classification
- **Entraînement** : Choix de l'algorithme, SMOTE
- **Évaluation** : Matrice de confusion, courbe ROC
- **Interprétabilité** : Importance des variables

### 📝 Page Conclusion
- Synthèse des résultats
- Recommandations métier
- Perspectives d'amélioration

---

## 🛠️ Installation

### Prérequis
- Python 3.9 ou supérieur
- pip (gestionnaire de packages Python)
- Navigateur web moderne

### Étapes d'Installation

```bash
# 1. Télécharger et extraire le projet
unzip hr_analytics_project.zip
cd hr_analytics_project

# 2. (Optionnel) Créer un environnement virtuel
python -m venv venv

# Sur Linux/Mac :
source venv/bin/activate

# Sur Windows :
venv\Scripts\activate

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Lancer l'application
streamlit run app.py
```

### Dépendances Principales

```
streamlit==1.28.0
pandas==2.1.0
numpy==1.24.0
scikit-learn==1.3.0
matplotlib==3.8.0
seaborn==0.13.0
plotly==5.18.0
prince==0.13.0
scipy==1.11.0
imbalanced-learn==0.11.0
```

---

## 🚀 Utilisation

### Lancement
```bash
streamlit run app.py
```
L'application s'ouvre automatiquement dans le navigateur à l'adresse : `http://localhost:8501`

### Chargement des Données
1. **Option 1** : Le dataset de démonstration est inclus dans `data/`
2. **Option 2** : Uploadez votre propre CSV via la barre latérale
3. **Option 3** : Téléchargez le dataset original depuis [Kaggle](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)

### Workflow Recommandé
1. 📊 **Explorer** les données dans l'onglet EDA
2. 🔬 **Analyser** avec l'AFDM (données mixtes)
3. 🎯 **Segmenter** avec K-Means (3-5 clusters)
4. 🤖 **Prédire** avec Random Forest + SMOTE
5. 📝 **Conclure** et formuler des recommandations

---

## 📂 Structure du Projet

```
hr_analytics_project/
│
├── 📄 app.py                    # Application Streamlit principale
├── 📄 requirements.txt          # Dépendances Python
├── 📄 README.md                 # Documentation (ce fichier)
├── 📄 download_dataset.py       # Script pour télécharger depuis Kaggle
│
├── 📁 data/
│   └── 📄 WA_Fn-UseC_-HR-Employee-Attrition.csv  # Dataset
│
└── 📁 rapport/                  # (À créer)
    ├── 📄 rapport_projet.pdf    # Rapport final
    └── 📄 presentation.pptx     # Support de présentation
```

---

## 📈 Résultats Attendus

### Analyse Factorielle (AFDM)
- **2-3 axes** expliquant 40-60% de l'inertie
- Identification d'une **opposition** entre employés satisfaits/insatisfaits
- Variables les plus contributrices : OverTime, JobSatisfaction, MonthlyIncome

### Clustering
- **3-5 clusters** distincts avec des profils interprétables
- Identification d'un cluster à **haut risque d'attrition** (>30%)
- Caractéristiques : jeunes, faible salaire, heures supplémentaires

### Classification
| Métrique | Objectif |
|----------|----------|
| F1-Score | > 0.70 |
| AUC-ROC | > 0.80 |
| Recall (Attrition=Yes) | > 0.60 |

### Variables Clés (Top 5)
1. **OverTime** - Heures supplémentaires
2. **MonthlyIncome** - Salaire mensuel
3. **Age** - Âge de l'employé
4. **YearsAtCompany** - Ancienneté
5. **JobSatisfaction** - Satisfaction au travail

---

## 💻 Technologies Utilisées

### Langages & Frameworks
| Technologie | Usage |
|-------------|-------|
| **Python 3.9+** | Langage principal |
| **Streamlit** | Interface web interactive |

### Librairies Data Science
| Librairie | Usage |
|-----------|-------|
| **Pandas** | Manipulation des données |
| **NumPy** | Calculs numériques |
| **Scikit-learn** | Machine Learning |
| **Prince** | Analyse factorielle (ACP, ACM, AFDM, AFC) |
| **Imbalanced-learn** | SMOTE pour le déséquilibre |

### Visualisation
| Librairie | Usage |
|-----------|-------|
| **Plotly** | Graphiques interactifs |
| **Matplotlib** | Graphiques statiques |
| **Seaborn** | Visualisations statistiques |

---

## 👥 Équipe

| Nom | Prénom | Contribution |
|-----|--------|--------------|
| ... | ... | EDA, Prétraitement des données |
| ... | ... | Analyse Factorielle, Interprétation |
| ... | ... | Clustering, Classification, Application |

---

## 📚 Références

### Bibliographie
- Escofier, B., & Pagès, J. (2008). *Analyses factorielles simples et multiples*. Dunod.
- Lebart, L., Piron, M., & Morineau, A. (2006). *Statistique exploratoire multidimensionnelle*. Dunod.
- Géron, A. (2019). *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow*. O'Reilly.

### Documentation Technique
- [Prince - Analyse Factorielle](https://github.com/MaxHalford/prince)
- [Scikit-learn](https://scikit-learn.org/stable/documentation.html)
- [Streamlit](https://docs.streamlit.io/)
- [Plotly](https://plotly.com/python/)
- [Imbalanced-learn](https://imbalanced-learn.org/)

### Dataset
- [IBM HR Analytics - Kaggle](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)

---

## 📜 Licence

Ce projet est réalisé dans un cadre académique pour le cours INFO0902 de l'Université de Reims Champagne-Ardenne.

---

<p align="center">
  <i>Projet réalisé dans le cadre du Master 2 IA & Data Science - URCA</i><br>
  <i>Année universitaire 2025-2026</i>
</p>