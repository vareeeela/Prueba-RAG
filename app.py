import json
import os
import re
import uuid
from datetime import datetime

import chromadb
import streamlit as st

from src.config import RUTA_BD
from src.generator import es_inyeccion_prompt, etiqueta_ubicacion, generar_respuesta, resumen_fuentes
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


_PATRON_SALUDO = re.compile(
    r"^(hola|buenos\s+(días|tardes|noches)|buenas|hey|hi|hello|qué\s+tal|cómo\s+est[aá]s?)[!?,. ]*$",
    re.IGNORECASE,
)


def es_saludo(texto: str) -> bool:
    return bool(_PATRON_SALUDO.match(texto.strip()))


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
    st.caption(f"{coleccion.count()} fragmentos indexados")


# ── Mostrar historial ──────────────────────────────────────────────────────

for msg in st.session_state.mensajes:
    with st.chat_message(msg["rol"]):
        st.write(msg["contenido"])
        if msg.get("fuentes"):
            st.caption(f"**Fuentes:** {msg['fuentes']}")
        if msg.get("chunks"):
            with st.expander(f"Ver {len(msg['chunks'])} fragmentos usados"):
                for i, (chunk, meta) in enumerate(zip(msg["chunks"], msg["metas"]), 1):
                    st.markdown(f"**[{i}] {etiqueta_ubicacion(meta)}**")
                    st.text(chunk)
                    if i < len(msg["chunks"]):
                        st.divider()


# ── Input ──────────────────────────────────────────────────────────────────

if pregunta := st.chat_input("Escribe tu pregunta..."):
    with st.chat_message("user"):
        st.write(pregunta)

    historial_previo = st.session_state.mensajes.copy()

    if not historial_previo:
        actualizar_titulo(st.session_state.conv_id, pregunta)

    st.session_state.mensajes.append({"rol": "user", "contenido": pregunta})

    with st.chat_message("assistant"):
        if es_saludo(pregunta):
            respuesta = "¡Hola! Soy lucIA, un asistente basado en tu documentación. ¿En qué puedo ayudarte?"
            st.write(respuesta)
            fuentes_str = ""
            chunks_guardados: list = []
            metas_guardadas: list = []
        elif es_inyeccion_prompt(pregunta):
            respuesta = (
                "Esta solicitud parece intentar modificar mi comportamiento o rol. "
                "Solo puedo responder preguntas sobre los documentos disponibles."
            )
            st.write(respuesta)
            fuentes_str = ""
            chunks_guardados = []
            metas_guardadas = []
        else:
            with st.spinner("Buscando en documentación..."):
                chunks, metas = buscar_contexto(coleccion, pregunta, historial=historial_previo)

            if not chunks:
                respuesta = "No encuentro fragmentos relevantes en la documentación. ¿Podrías reformular la pregunta o ser más específico?"
                st.write(respuesta)
                fuentes_str = ""
                chunks_guardados = []
                metas_guardadas = []
            else:
                respuesta = st.write_stream(
                    generar_respuesta(chunks, metas, pregunta, historial=historial_previo)
                )
                fuentes_str = resumen_fuentes(metas)
                st.caption(f"**Fuentes:** {fuentes_str}")
                with st.expander(f"Ver {len(chunks)} fragmentos usados"):
                    for i, (chunk, meta) in enumerate(zip(chunks, metas), 1):
                        st.markdown(f"**[{i}] {etiqueta_ubicacion(meta)}**")
                        st.text(chunk)
                        if i < len(chunks):
                            st.divider()
                chunks_guardados = chunks
                metas_guardadas = metas

    st.session_state.mensajes.append({
        "rol": "assistant",
        "contenido": respuesta,
        "fuentes": fuentes_str,
        "chunks": chunks_guardados,
        "metas": metas_guardadas,
    })
    guardar_mensajes(st.session_state.conv_id, st.session_state.mensajes)
