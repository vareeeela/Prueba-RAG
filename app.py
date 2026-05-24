import json
import os

import chromadb
import streamlit as st

from src.config import RUTA_BD
from src.generator import generar_respuesta, resumen_fuentes
from src.indexer import indexar_documentos, obtener_coleccion
from src.retriever import buscar_contexto

RUTA_HISTORIAL = os.path.join(RUTA_BD, "historial.json")


def cargar_historial() -> list[dict]:
    if os.path.exists(RUTA_HISTORIAL):
        with open(RUTA_HISTORIAL, encoding="utf-8") as f:
            return json.load(f)
    return []


def guardar_historial(mensajes: list[dict]) -> None:
    os.makedirs(RUTA_BD, exist_ok=True)
    with open(RUTA_HISTORIAL, "w", encoding="utf-8") as f:
        json.dump(mensajes, f, ensure_ascii=False, indent=2)


st.set_page_config(page_title="lucIA", page_icon="ฅᨐฅ", layout="centered")
st.title("lucIA ฅᨐฅ")


@st.cache_resource(show_spinner="Cargando base de conocimiento...")
def cargar_sistema() -> chromadb.Collection:
    cliente = chromadb.PersistentClient(path=RUTA_BD)
    coleccion = obtener_coleccion(cliente)
    indexar_documentos(coleccion)
    return coleccion


coleccion = cargar_sistema()

if "mensajes" not in st.session_state:
    st.session_state.mensajes = cargar_historial()

# Sidebar
with st.sidebar:
    st.header("Conversación")
    if st.button("🗑️ Limpiar historial", use_container_width=True):
        st.session_state.mensajes = []
        guardar_historial([])
        st.rerun()

# Mostrar historial
for msg in st.session_state.mensajes:
    with st.chat_message(msg["rol"]):
        st.write(msg["contenido"])
        if msg.get("fuentes"):
            st.caption(f"**Fuentes:** {msg['fuentes']}")

# Input
if pregunta := st.chat_input("Escribe tu pregunta..."):
    with st.chat_message("user"):
        st.write(pregunta)

    # Historial previo que verá el LLM (antes de añadir la pregunta actual)
    historial_previo = st.session_state.mensajes.copy()
    st.session_state.mensajes.append({"rol": "user", "contenido": pregunta})

    with st.chat_message("assistant"):
        with st.spinner("Buscando en documentación..."):
            chunks, metas = buscar_contexto(coleccion, pregunta)

        if not chunks:
            respuesta = "No se encontraron fragmentos relevantes en la documentación."
            st.write(respuesta)
            fuentes_str = ""
        else:
            respuesta = st.write_stream(
                generar_respuesta(chunks, metas, pregunta, historial=historial_previo)
            )
            fuentes_str = resumen_fuentes(metas)
            st.caption(f"**Fuentes:** {fuentes_str}")

    st.session_state.mensajes.append({
        "rol": "assistant",
        "contenido": respuesta,
        "fuentes": fuentes_str,
    })
    guardar_historial(st.session_state.mensajes)
