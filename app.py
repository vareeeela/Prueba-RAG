import json
import os
import uuid
from datetime import datetime

import chromadb
import streamlit as st

from src.config import RUTA_BD
from src.generator import es_inyeccion_prompt, generar_respuesta, resumen_fuentes, respuesta_identidad
from src.indexer import indexar_documentos, obtener_coleccion
from src.retriever import buscar_contexto

RUTA_CONVERSACIONES = os.path.join(RUTA_BD, "conversations")
RUTA_INDICE = os.path.join(RUTA_BD, "conversations_index.json")
RUTA_HISTORIAL_LEGACY = os.path.join(RUTA_BD, "historial.json")


# ── Gestión de conversaciones ──────────────────────────────────────────────

def cargar_indice() -> list[dict]:
    if os.path.exists(RUTA_INDICE):
        with open(RUTA_INDICE, encoding="utf-8") as f:
            return json.load(f)
    return []


def guardar_indice(indice: list[dict]) -> None:
    os.makedirs(RUTA_BD, exist_ok=True)
    with open(RUTA_INDICE, "w", encoding="utf-8") as f:
        json.dump(indice, f, ensure_ascii=False, indent=2)


def cargar_mensajes(conv_id: str) -> list[dict]:
    ruta = os.path.join(RUTA_CONVERSACIONES, f"{conv_id}.json")
    if os.path.exists(ruta):
        with open(ruta, encoding="utf-8") as f:
            return json.load(f)
    return []


def guardar_mensajes(conv_id: str, mensajes: list[dict]) -> None:
    os.makedirs(RUTA_CONVERSACIONES, exist_ok=True)
    with open(os.path.join(RUTA_CONVERSACIONES, f"{conv_id}.json"), "w", encoding="utf-8") as f:
        json.dump(mensajes, f, ensure_ascii=False, indent=2)


def crear_conversacion(titulo: str = "Nueva conversación") -> str:
    conv_id = uuid.uuid4().hex[:8]
    ahora = datetime.now().isoformat()
    indice = cargar_indice()
    indice.insert(0, {"id": conv_id, "titulo": titulo, "created_at": ahora, "updated_at": ahora})
    guardar_indice(indice)
    guardar_mensajes(conv_id, [])
    return conv_id


def actualizar_titulo(conv_id: str, primera_pregunta: str) -> None:
    titulo = primera_pregunta[:50] + ("…" if len(primera_pregunta) > 50 else "")
    indice = cargar_indice()
    for conv in indice:
        if conv["id"] == conv_id:
            conv["titulo"] = titulo
            conv["updated_at"] = datetime.now().isoformat()
            break
    guardar_indice(indice)


def borrar_conversacion(conv_id: str) -> None:
    ruta = os.path.join(RUTA_CONVERSACIONES, f"{conv_id}.json")
    if os.path.exists(ruta):
        os.remove(ruta)
    guardar_indice([c for c in cargar_indice() if c["id"] != conv_id])


def migrar_historial_legacy() -> None:
    """Migra historial.json antiguo a una conversación si aún no existe el índice."""
    if not os.path.exists(RUTA_HISTORIAL_LEGACY) or os.path.exists(RUTA_INDICE):
        return
    with open(RUTA_HISTORIAL_LEGACY, encoding="utf-8") as f:
        mensajes = json.load(f)
    if mensajes:
        conv_id = crear_conversacion("Conversación anterior")
        guardar_mensajes(conv_id, mensajes)


# ── Inicialización ─────────────────────────────────────────────────────────

migrar_historial_legacy()

st.set_page_config(page_title="lucIA", page_icon="👻", layout="centered")
st.title("lucIA ฅᨐฅ")

st.markdown("""
<style>
/* Conversaciones de la sidebar como ítems de nav */
section[data-testid="stSidebar"] .stButton > button {
    border: none !important;
    background: transparent !important;
    text-align: start !important;
    justify-content: flex-start !important;
    padding: 0.3rem 0.5rem !important;
    border-radius: 6px !important;
    font-size: 0.85rem !important;
    color: inherit !important;
    transition: background 0.15s;
}
section[data-testid="stSidebar"] .stButton > button:hover {
    background: rgba(128,128,128,0.12) !important;
}
/* Botón de nueva conversación */
section[data-testid="stSidebar"] > div > div:first-child .stButton > button {
    background: rgba(128,128,128,0.08) !important;
    font-weight: 500 !important;
    margin-block-end: 0.25rem !important;
}
/* Botón de borrar: más pequeño y sutil */
section[data-testid="stSidebar"] [data-testid="column"]:last-child .stButton > button {
    color: rgba(128,128,128,0.6) !important;
    font-size: 0.8rem !important;
    padding: 0.3rem 0.2rem !important;
}
section[data-testid="stSidebar"] [data-testid="column"]:last-child .stButton > button:hover {
    color: #e05c5c !important;
    background: rgba(224,92,92,0.08) !important;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Cargando base de conocimiento...")
def cargar_sistema() -> chromadb.Collection:
    cliente = chromadb.PersistentClient(path=RUTA_BD)
    coleccion = obtener_coleccion(cliente)
    indexar_documentos(coleccion)
    return coleccion


coleccion = cargar_sistema()

if "conv_id" not in st.session_state:
    indice = cargar_indice()
    if indice:
        st.session_state.conv_id = indice[0]["id"]
        st.session_state.mensajes = cargar_mensajes(indice[0]["id"])
    else:
        st.session_state.conv_id = crear_conversacion()
        st.session_state.mensajes = []


# ── Sidebar ────────────────────────────────────────────────────────────────

with st.sidebar:
    if st.button("＋ Nueva conversación", use_container_width=True):
        st.session_state.conv_id = crear_conversacion()
        st.session_state.mensajes = []
        st.rerun()

    st.divider()

    indice = cargar_indice()
    for conv in indice:
        es_activa = conv["id"] == st.session_state.conv_id
        col1, col2 = st.columns([5, 1])
        with col1:
            etiqueta = f"**{conv['titulo']}**" if es_activa else conv["titulo"]
            if st.button(etiqueta, key=f"sel_{conv['id']}", use_container_width=True):
                st.session_state.conv_id = conv["id"]
                st.session_state.mensajes = cargar_mensajes(conv["id"])
                st.rerun()
        with col2:
            if st.button("🗑", key=f"del_{conv['id']}"):
                borrar_conversacion(conv["id"])
                indice_nuevo = cargar_indice()
                if indice_nuevo:
                    st.session_state.conv_id = indice_nuevo[0]["id"]
                    st.session_state.mensajes = cargar_mensajes(indice_nuevo[0]["id"])
                else:
                    st.session_state.conv_id = crear_conversacion()
                    st.session_state.mensajes = []
                st.rerun()

    st.divider()


# ── Mostrar historial ──────────────────────────────────────────────────────

for msg in st.session_state.mensajes:
    with st.chat_message(msg["rol"]):
        st.write(msg["contenido"])
        if msg.get("fuente_citada"):
            st.caption(msg["fuente_citada"])


# ── Input ──────────────────────────────────────────────────────────────────

if pregunta := st.chat_input("Escribe tu pregunta..."):
    with st.chat_message("user"):
        st.write(pregunta)

    historial_previo = st.session_state.mensajes.copy()

    if not historial_previo:
        actualizar_titulo(st.session_state.conv_id, pregunta)

    st.session_state.mensajes.append({"rol": "user", "contenido": pregunta})

    fuente_citada = ""
    with st.chat_message("assistant"):
        resp_fija = respuesta_identidad(pregunta)
        if resp_fija:
            respuesta = resp_fija
            st.markdown(respuesta)
        elif es_inyeccion_prompt(pregunta):
            respuesta = (
                "Esta solicitud parece intentar modificar mi comportamiento o rol. "
                "Solo puedo responder preguntas sobre los documentos disponibles."
            )
            st.write(respuesta)
        else:
            with st.spinner("Buscando en documentación..."):
                chunks, metas = buscar_contexto(coleccion, pregunta, historial=historial_previo)

            if not chunks:
                respuesta = "Esta consulta no está cubierta por los documentos disponibles."
                st.write(respuesta)
            else:
                placeholder = st.empty()
                respuesta = ""
                for token in generar_respuesta(chunks, metas, pregunta, historial=historial_previo):
                    respuesta += token
                    placeholder.markdown(respuesta)
                fuente_citada = resumen_fuentes(metas)
                if fuente_citada:
                    st.caption(fuente_citada)

    st.session_state.mensajes.append({
        "rol": "assistant",
        "contenido": respuesta,
        "fuente_citada": fuente_citada,
    })
    guardar_mensajes(st.session_state.conv_id, st.session_state.mensajes)
