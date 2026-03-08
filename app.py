import streamlit as st
import joblib

# Título de la aplicación
st.title("Clasificador de Textos ODS")
st.write("Ingresa un texto y el modelo predecirá a qué Objetivo de Desarrollo Sostenible (ODS) pertenece.")

# Carga de modelos sin manejo complejo de errores (nivel estudiante)
pipeline = joblib.load('modelo_ods.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# Diccionario con los nombres oficiales de los 17 ODS
nombres_ods = {
    1: "Fin de la pobreza",
    2: "Hambre cero",
    3: "Salud y bienestar",
    4: "Educación de calidad",
    5: "Igualdad de género",
    6: "Agua limpia y saneamiento",
    7: "Energía asequible y no contaminante",
    8: "Trabajo decente y crecimiento económico",
    9: "Industria, innovación e infraestructura",
    10: "Reducción de las desigualdades",
    11: "Ciudades y comunidades sostenibles",
    12: "Producción y consumo responsables",
    13: "Acción por el clima",
    14: "Vida submarina",
    15: "Vida de ecosistemas terrestres",
    16: "Paz, justicia e instituciones sólidas",
    17: "Alianzas para lograr los objetivos"
}

# Campo de texto para el usuario
texto_usuario = st.text_area("Texto a clasificar:", height=150)

# Botón de predicción
if st.button("Predecir ODS"):
    if texto_usuario != "":
        # Hacer la predicción
        prediccion_num = pipeline.predict([texto_usuario])
        
        # Convertir la predicción (ej. 4) tomando el entero real
        ods_numero = int(label_encoder.inverse_transform(prediccion_num)[0])
        ods_texto = nombres_ods.get(ods_numero, "ODS Desconocido")
        
        # Mostrar el resultado
        st.write("---")
        st.write(f"ODS {ods_numero}: {ods_texto}")
    else:
        st.write("Por favor ingresa un texto primero.")
