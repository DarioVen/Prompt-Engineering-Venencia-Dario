import streamlit as st 
from transformers import pipeline
import os 

# Configuración de la página 
st.set_page_config(page_title="Analizador de Correos", page_icon="📧", layout="centered") 

# Inicializar el modelo de Hugging Face
@st.cache_resource
def load_model():
    return pipeline("summarization", model="facebook/bart-base")

generator = load_model()

st.markdown("## 📧 Analizador de Correos") 
st.markdown("Ingresa el contenido del correo electrónico para analizarlo y extraer información clave y tareas pendientes.") 

# Input del usuario 
st.markdown("### 📥 Información del Correo")
email_content = st.text_area("✉️ Contenido del correo", height=200, placeholder="Pega aquí el contenido del correo electrónico...") 

# Botón 
if st.button("Analizar"): 
    if not email_content.strip(): 
        st.warning("Por favor, ingresa el contenido del correo antes de analizar.") 
    else: 
        with st.spinner("Analizando el correo..."): 
            try:
                # Truncar y preparar el contenido
                words = email_content.split()
                truncated_content = ' '.join(words[:150])
                
                # Generar análisis con prompt mejorado
                summary_results = generator(
                    f"Analiza este correo electrónico y extrae los puntos más importantes: {truncated_content}",
                    max_length=150,
                    min_length=50,
                    length_penalty=2.0,
                    num_beams=4,
                    early_stopping=True
                )
                
                # Generar detalles con prompt mejorado
                detail_results = generator(
                    f"Lista las tareas pendientes, fechas importantes y acciones requeridas de este correo: {truncated_content}",
                    max_length=200,
                    min_length=75,
                    length_penalty=2.5,
                    num_beams=5,
                    early_stopping=True
                )
                
                if summary_results and detail_results:
                    summary = summary_results[0]['summary_text'].strip()
                    details = detail_results[0]['summary_text'].strip()
                    
                    st.markdown("### 📋 Resumen General:") 
                    st.write(summary)
                    
                    st.markdown("### 📌 Detalles y Tareas Específicas:")
                    st.write(details)
                    
                    if len(words) > 150:
                        st.warning("⚠️ Nota: El correo fue truncado debido a su longitud. El análisis se realizó sobre las primeras 150 palabras.")
                else:
                    st.error("No se pudo generar un análisis válido. Por favor, intenta de nuevo.")
                    
            except Exception as e: 
                st.error(f"Error al analizar el correo: {str(e)}") 

# Agregar información de pie de página
st.markdown("---")
st.markdown("💡 **Nota:** Esta herramienta utiliza IA de Hugging Face para analizar correos electrónicos y extraer información relevante.")