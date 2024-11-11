import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import cv2
import os
import plotly.graph_objects as go


def set_custom_style():
    """
    Définit le style personnalisé de l'interface
    """
    st.markdown("""
    <style>
        /* Style général de la page */
        .stApp {
            background-image: linear-gradient(to bottom, rgba(255,255,255,0.9) 0%,rgba(255,255,255,0.9) 100%), url('https://img.freepik.com/photos-gratuite/texture-sol-brun-fonce-vue-dessus_87394-6862.jpg');
            background-size: cover;
            background-attachment: fixed;
        }
        
        /* Style pour les titres */
        h1 {
            color: #2e7d32;
            text-align: center;
            padding: 20px;
            border-bottom: 2px solid #2e7d32;
            margin-bottom: 30px;
        }
        
        /* Style pour les sections */
        .css-1d391kg {
            padding: 20px;
            border-radius: 10px;
            background-color: rgba(255, 255, 255, 0.9);
            margin: 10px 0;
        }
        
        /* Style pour les cartes */
        .recommendation-section {
            background-color: rgba(255, 255, 255, 0.95) !important;
            border-left: 5px solid #2e7d32;
            transition: transform 0.2s;
        }
        
        .recommendation-section:hover {
            transform: translateX(5px);
        }
        
        /* Style pour le graphique */
        .soil-chart {
            background-color: rgba(255, 255, 255, 0.95);
            padding: 15px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        /* Style pour les instructions */
        .instructions {
            background-color: rgba(46, 125, 50, 0.1);
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        
        /* Style pour le bouton upload */
        .css-1offfwp {
            border-color: #2e7d32 !important;
            color: #2e7d32 !important;
        }
        
        /* Style pour les alertes */
        .stAlert {
            background-color: rgba(255, 255, 255, 0.95) !important;
        }
    </style>
    """, unsafe_allow_html=True)

def display_probability_chart(prediction, soil_classes):
    """
    Affiche un graphique des probabilités pour chaque type de sol
    """
    
    
    # Préparer les données
    probabilities = [float(p) * 100 for p in prediction[0]]
    colors = ['#2e7d32' if prob == max(probabilities) else '#90a4ae' for prob in probabilities]
    
    # Créer le graphique
    fig = go.Figure(data=[
        go.Bar(
            x=soil_classes,
            y=probabilities,
            marker_color=colors,
            text=[f'{prob:.1f}%' for prob in probabilities],
            textposition='auto',
        )
    ])
    
    # Personnaliser le graphique
    fig.update_layout(
        title={
            'text': 'Probabilités par type de sol',
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        xaxis_title="Types de sol",
        yaxis_title="Probabilité (%)",
        yaxis_range=[0, 100],
        template='plotly_white',
        plot_bgcolor='rgba(255,255,255,0.9)',
        paper_bgcolor='rgba(0,0,0,0)',
    )
    
    # Afficher le graphique dans Streamlit
    st.plotly_chart(fig, use_container_width=True)

# Configuration de la page Streamlit
st.set_page_config(
    page_title="Analyse de Sol et Recommandations Agricoles",
    page_icon="🌱",
    layout="centered"
)

def analyze_image_features(image):
    """
    Analyse les caractéristiques de l'image en utilisant OpenCV
    """
    try:
        # Convertir l'image PIL en array numpy
        img_array = np.array(image)
        
        # Convertir en niveaux de gris
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
            
        # Calculer les caractéristiques de texture avec OpenCV
        # 1. Calculer la matrice de cooccurrence manuelle
        def calculate_glcm_features(img, distance=1):
            height, width = img.shape
            glcm = np.zeros((256, 256))
            
            for i in range(height-distance):
                for j in range(width-distance):
                    i_intensity = img[i, j]
                    j_intensity = img[i+distance, j+distance]
                    glcm[i_intensity, j_intensity] += 1
            
            # Normaliser la matrice
            glcm = glcm / glcm.sum()
            
            # Calculer les caractéristiques
            contrast = 0
            homogeneity = 0
            energy = 0
            
            for i in range(256):
                for j in range(256):
                    contrast += glcm[i,j] * (i-j)**2
                    homogeneity += glcm[i,j] / (1 + abs(i-j))
                    energy += glcm[i,j]**2
                    
            return contrast, homogeneity, energy
        
        # Calculer les caractéristiques
        contrast, homogeneity, energy = calculate_glcm_features(gray)
        
        # Calculer d'autres caractéristiques avec OpenCV
        mean_val = cv2.mean(gray)[0]
        std_dev = cv2.meanStdDev(gray)[1][0][0]
        
        # Vérifier si les caractéristiques correspondent à celles typiques d'un sol
        is_likely_soil = (
            contrast > 100 and  # Ajusté pour OpenCV
            homogeneity > 0.1 and
            energy < 0.1 and
            std_dev > 10  # Les sols naturels ont généralement une certaine variation
        )
        
        return is_likely_soil, {
            'contrast': contrast,
            'homogeneity': homogeneity,
            'energy': energy,
            'std_dev': std_dev
        }
        
    except Exception as e:
        st.error(f"Erreur lors de l'analyse des caractéristiques : {str(e)}")
        return False, {}

def preprocess_image(image):
    try:
        # Vérifier d'abord si l'image ressemble à un sol
        is_soil, features = analyze_image_features(image)
        
        if not is_soil:
            st.error("⚠️ Cette image ne semble pas être une photo de sol. Veuillez uploader une photo de sol naturel.")
            st.write("L'image ne présente pas les caractéristiques typiques d'un sol :")
            st.write(f"- Contraste : {features.get('contrast', 0):.2f}")
            st.write(f"- Homogénéité : {features.get('homogeneity', 0):.2f}")
            st.write(f"- Énergie : {features.get('energy', 0):.2f}")
            st.write(f"- Écart-type : {features.get('std_dev', 0):.2f}")
            return None

        # Redimensionner l'image
        img = image.resize((260, 260))
        img_array = np.array(img)
        
        # Vérification supplémentaire des caractéristiques visuelles
        if len(img_array.shape) != 3 or img_array.shape[2] != 3:
            st.error("⚠️ L'image doit être en couleur (RGB).")
            return None
            
        # Normalisation
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
        
    except Exception as e:
        st.error(f"Erreur lors du prétraitement de l'image: {str(e)}")
        return None


def get_crop_recommendations(soil_type):
    recommendations = {
        'noir': {
            'fertilite': "Très fertile - Riche en matière organique et en nutriments",
            'cultures': [
                "Maïs - Rendement optimal grâce à la richesse en nutriments",
                "Coton - Excellent pour les sols profonds et riches",
                "Sorgho - Bonne adaptation aux sols noirs",
                "Tournesol - Profite bien de la richesse du sol"
            ],
            'conseils': [
                "Maintenir un bon drainage",
                "Éviter le travail du sol en conditions très humides",
                "Rotation des cultures recommandée"
            ]
        },
        'argileux': {
            'fertilite': "Moyennement fertile - Bonne rétention d'eau et de nutriments",
            'cultures': [
                "Riz - Parfait pour la rétention d'eau",
                "Blé - Bien adapté aux sols argileux",
                "Soja - Bonne croissance en sol argileux",
                "Légumineuses - Contribuent à améliorer la structure du sol"
            ],
            'conseils': [
                "Améliorer la structure du sol avec de la matière organique",
                "Travailler le sol au bon moment (ni trop sec, ni trop humide)",
                "Prévoir un système de drainage efficace"
            ]
        },
        'rouge': {
            'fertilite': "Fertilité moyenne - Riche en fer mais peut manquer de nutriments",
            'cultures': [
                "Manioc - Bien adapté aux sols plus pauvres",
                "Arachide - Bonne adaptation aux sols rouges",
                "Mil - Résistant et adapté à ces sols",
                "Patate douce - Bonne croissance dans les sols rouges"
            ],
            'conseils': [
                "Enrichir le sol en matière organique",
                "Pratiquer le paillage pour conserver l'humidité",
                "Utiliser des engrais adaptés pour compenser les carences"
            ]
        },
        'alluvial': {
            'fertilite': "Très fertile - Sol riche et bien équilibré",
            'cultures': [
                "Légumes variés - Excellent pour le maraîchage",
                "Canne à sucre - Profite bien de l'humidité",
                "Banane - Croissance optimale",
                "Fruits et légumes - Grande variété possible"
            ],
            'conseils': [
                "Maintenir le niveau de matière organique",
                "Surveiller le niveau d'eau (risque d'inondation)",
                "Rotation des cultures pour optimiser la fertilité"
            ]
        }
    }
    return recommendations.get(soil_type, None)

@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model('model_2.h5')
        return model
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle: {str(e)}")
        return None

def main():
    # Appliquer le style personnalisé
    set_custom_style()
    
    st.title("🌱 Analyse de Sol et Recommandations Agricoles")
    
    # Ajouter la classe CSS aux instructions
    st.markdown("""
    <div class='instructions'>
        <h3>📋 Instructions :</h3>
        <ol>
            <li>Uploadez une photo de sol naturel</li>
            <li>La photo doit être claire et bien éclairée</li>
            <li>Évitez les photos de sols artificiels (carrelage, béton, etc.)</li>
            <li>Évitez les photos de visages ou d'autres objets</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)

    model = load_model()
    if model is None:
        st.error("Impossible de charger le modèle. Veuillez réessayer.")
        return

    uploaded_file = st.file_uploader("Choisissez une photo de sol", type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        try:
                # Créer deux grandes colonnes avec un ratio égal
                col1, col2 = st.columns([1, 1])
        
                with col1:
                    # Afficher l'image avec une plus grande taille
                    image = Image.open(uploaded_file)
                    st.image(image, caption="Image chargée avec succès", use_container_width=True)

                processed_image = preprocess_image(image)
                if processed_image is None:
                    return
                
                prediction = model.predict(processed_image)
                class_index = np.argmax(prediction[0])
                confidence = prediction[0][class_index] * 100

                if confidence < 70:
                    st.error("""
                             ⚠️ La prédiction n'est pas assez fiable. 
                                Cela pourrait indiquer que l'image n'est pas une photo de sol appropriée.
                                Veuillez uploader une photo plus claire d'un sol naturel.
                            """)
                    return
        
                soil_classes = ['alluvial', 'argileux', 'noir', 'rouge']
                predicted_soil = soil_classes[class_index]

                with col2:
                            # Créer un conteneur pour regrouper les résultats et le graphique
                    with st.container():
                        # Afficher les résultats de l'analyse
                            st.markdown(f"""
                <div style='background-color: rgba(255, 255, 255, 0.9); padding: 20px; border-radius: 10px; margin-bottom: 20px;'>
                    <h3 style='color: #2e7d32; margin-top: 0;'>Résultats de l'analyse</h3>
                    <p style='font-size: 18px;'>Type de sol détecté : <strong>Sol {predicted_soil.capitalize()}</strong></p>
                    <p>Confiance de la prédiction : <strong>{confidence:.2f}%</strong></p>
                </div>
                """, unsafe_allow_html=True)
                
                # Afficher le graphique juste en dessous
                display_probability_chart(prediction, soil_classes)

                # Le reste du code pour les recommandations reste inchangé
                recommendations = get_crop_recommendations(predicted_soil)
                if recommendations:
                    st.markdown("""
            <style>
                .recommendation-grid {
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                    gap: 20px;
                    margin-top: 20px;
                }
                .recommendation-card {
                    background-color: rgba(255, 255, 255, 0.95);
                    padding: 20px;
                    border-radius: 10px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    border-left: 5px solid #2e7d32;
                }
            </style>
            <div class='recommendation-grid'>
            """, unsafe_allow_html=True)
                # Fertilité
                st.markdown(f"""
                <div class='recommendation-card'>
                    <h3>📊 Analyse de fertilité</h3>
                    <p>{recommendations['fertilite']}</p>
                </div>
                """, unsafe_allow_html=True)

                # Cultures recommandées
                cultures_html = "<div class='recommendation-card'><h3>🌾 Cultures recommandées</h3><ul>"
                for culture in recommendations['cultures']:
                    cultures_html += f"<li>{culture}</li>"
                cultures_html += "</ul></div>"
                st.markdown(cultures_html, unsafe_allow_html=True)

                # Conseils
                conseils_html = "<div class='recommendation-card'><h3>💡 Conseils de gestion</h3><ul>"
                for conseil in recommendations['conseils']:
                    conseils_html += f"<li>{conseil}</li>"
                conseils_html += "</ul></div>"
                st.markdown(conseils_html, unsafe_allow_html=True)

                st.markdown("</div>", unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Une erreur s'est produite lors de l'analyse : {str(e)}")

if __name__ == "__main__":
    main()