#!/usr/bin/env python3
"""
Script pour télécharger le dataset IBM HR Analytics depuis Kaggle.
Nécessite d'avoir configuré l'API Kaggle avec vos credentials.

Instructions:
1. Créez un compte Kaggle si vous n'en avez pas
2. Allez dans Account > Create New API Token
3. Placez kaggle.json dans ~/.kaggle/
4. Exécutez ce script

Ou téléchargez manuellement depuis:
https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset
"""

import os
import sys

def download_dataset():
    """Télécharge le dataset depuis Kaggle"""
    try:
        import kaggle
        
        # Créer le dossier data s'il n'existe pas
        os.makedirs('data', exist_ok=True)
        
        # Télécharger le dataset
        kaggle.api.dataset_download_files(
            'pavansubhasht/ibm-hr-analytics-attrition-dataset',
            path='data',
            unzip=True
        )
        
        print("✅ Dataset téléchargé avec succès dans le dossier 'data/'")
        
    except ImportError:
        print("❌ Le package kaggle n'est pas installé.")
        print("   Installez-le avec: pip install kaggle")
        print("\n📥 Téléchargement manuel:")
        print("   1. Allez sur: https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset")
        print("   2. Téléchargez le fichier CSV")
        print("   3. Placez-le dans le dossier 'data/'")
        
    except Exception as e:
        print(f"❌ Erreur: {e}")
        print("\n📥 Téléchargement manuel:")
        print("   1. Allez sur: https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset")
        print("   2. Téléchargez le fichier CSV")
        print("   3. Placez-le dans le dossier 'data/'")


if __name__ == "__main__":
    download_dataset()
