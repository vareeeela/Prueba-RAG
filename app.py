import json
import os
import uuid
from datetime import datetime

import chromadb
import streamlit as st

from src.config import RUTA_BD
from src.generator import (
    es_inyeccion_prompt,
    extraer_metas_citadas,
    generar_respuesta,
    reemplazar_citas,
    resumen_fuentes,
    respuesta_identidad,
)
from src.indexer import indexar_documentos, obtener_coleccion
from src.retriever import buscar_contexto, detectar_intencion

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
    with open(
        os.path.join(RUTA_CONVERSACIONES, f"{conv_id}.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(mensajes, f, ensure_ascii=False, indent=2)


def crear_conversacion(titulo: str = "Nueva conversación") -> str:
    conv_id = uuid.uuid4().hex[:8]
    ahora = datetime.now().isoformat()
    indice = cargar_indice()
    indice.insert(0, {"id": conv_id, "titulo": titulo, "created_at": ahora, "updated_at": ahora})
    guardar_indice(indice)
    guardar_mensajes(conv_id, [])
    return conv_id


def _titulo_corto(pregunta: str) -> str:
    """Genera un título de 3-5 palabras con el modelo LLM."""
    from src.config import GROQ_API_KEY, MODO, MODELO_GROQ, MODELO_LLM
    prompt = (
        "Resume en 3-5 palabras el tema de esta pregunta como título de conversación. "
        "Solo las palabras clave, sin puntuación ni explicación.\n\n"
        f"Pregunta: {pregunta}\nTítulo:"
    )
    try:
        if MODO == "local":
            import ollama
            resp = ollama.chat(
                model=MODELO_LLM,
                messages=[{"role": "user", "content": prompt}],
                options={"num_predict": 15},
            )
            titulo = resp["message"]["content"].strip()
        else:
            from groq import Groq
            resp = Groq(api_key=GROQ_API_KEY).chat.completions.create(
                model=MODELO_GROQ,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=15,
                temperature=0.2,
            )
            titulo = resp.choices[0].message.content.strip()
        titulo = titulo.strip('"\'«»').strip()
        return titulo[:60] if titulo else pregunta[:40]
    except Exception:
        return pregunta[:40]


def actualizar_titulo(conv_id: str, primera_pregunta: str) -> None:
    titulo = _titulo_corto(primera_pregunta)
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
    if not os.path.exists(RUTA_HISTORIAL_LEGACY) or os.path.exists(RUTA_INDICE):
        return
    with open(RUTA_HISTORIAL_LEGACY, encoding="utf-8") as f:
        mensajes = json.load(f)
    if mensajes:
        conv_id = crear_conversacion("Conversación anterior")
        guardar_mensajes(conv_id, mensajes)


migrar_historial_legacy()

# ── Configuración de la página ──────────────────────────────────────────────

st.set_page_config(
    page_title="lucIA · ISO 27001/27002",
    page_icon="🔒",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
/* ─── Layout ─── */
.block-container {
    padding-top: 1.8rem !important;
    padding-bottom: 0.5rem !important;
    max-width: 800px !important;
}

/* ─── Sidebar oscura ─── */
[data-testid="stSidebar"] {
    background-color: #0f172a !important;
}
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label {
    color: #94a3b8 !important;
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] strong {
    color: #e2e8f0 !important;
}
[data-testid="stSidebar"] hr {
    border-color: rgba(255,255,255,0.08) !important;
    margin: 0.5rem 0 !important;
}

/* ─── Botones sidebar ─── */
[data-testid="stSidebar"] .stButton > button {
    border: none !important;
    background: transparent !important;
    text-align: left !important;
    justify-content: flex-start !important;
    padding: 0.4rem 0.7rem !important;
    border-radius: 7px !important;
    font-size: 0.82rem !important;
    color: #cbd5e1 !important;
    width: 100% !important;
    transition: background 0.1s ease;
    white-space: nowrap !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: rgba(255,255,255,0.07) !important;
    color: #f1f5f9 !important;
}

/* Botón borrar: pequeño y sutil */
[data-testid="stSidebar"] [data-testid="column"]:last-child .stButton > button {
    color: rgba(148,163,184,0.4) !important;
    font-size: 0.75rem !important;
    padding: 0.35rem 0.25rem !important;
    text-align: center !important;
    justify-content: center !important;
}
[data-testid="stSidebar"] [data-testid="column"]:last-child .stButton > button:hover {
    color: #f87171 !important;
    background: rgba(248,113,113,0.1) !important;
}

/* ─── Cabecera principal ─── */
.lucia-header {
    display: flex;
    align-items: baseline;
    gap: 0.5rem;
    margin-bottom: 0.1rem;
}

/* ─── Estado vacío ─── */
.empty-state {
    text-align: center;
    padding: 2.5rem 1.5rem;
    color: #94a3b8;
    border: 1.5px dashed #e2e8f0;
    border-radius: 14px;
    margin: 1rem 0 1.5rem;
}
.empty-state .icon { font-size: 2rem; margin-bottom: 0.6rem; }
.empty-state strong { color: #64748b; font-size: 0.95rem; display: block; margin-bottom: 0.4rem; }
.empty-state ul {
    list-style: none;
    padding: 0;
    margin: 0.8rem 0 0;
    text-align: left;
    display: inline-block;
}
.empty-state ul li {
    font-size: 0.82rem;
    padding: 0.25rem 0;
    color: #94a3b8;
}
.empty-state ul li::before { content: "→  "; color: #cbd5e1; }

/* ─── Mensajes de chat ─── */
[data-testid="stChatMessage"] { padding: 0.6rem 0 !important; }

/* ─── Fuentes ─── */
[data-testid="stCaptionContainer"] { opacity: 0.55; font-size: 0.72rem !important; }

/* ─── Input ─── */
[data-testid="stChatInput"] textarea {
    font-size: 0.95rem !important;
    border-radius: 10px !important;
}
</style>
""", unsafe_allow_html=True)


# ── Carga del sistema ────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Cargando base de conocimiento…")
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


# ── Sidebar ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### lucIA")
    st.caption("ISO 27001 · ISO 27002")
    st.divider()

    if st.button("＋  Nueva conversación", use_container_width=True, type="secondary"):
        st.session_state.conv_id = crear_conversacion()
        st.session_state.mensajes = []
        st.rerun()

    st.divider()
    st.caption("Historial")

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
            if st.button("✕", key=f"del_{conv['id']}"):
                borrar_conversacion(conv["id"])
                indice_nuevo = cargar_indice()
                if indice_nuevo:
                    st.session_state.conv_id = indice_nuevo[0]["id"]
                    st.session_state.mensajes = cargar_mensajes(indice_nuevo[0]["id"])
                else:
                    st.session_state.conv_id = crear_conversacion()
                    st.session_state.mensajes = []
                st.rerun()


# ── Área principal ───────────────────────────────────────────────────────────

st.markdown("### lucIA ฅᨐฅ")
st.caption("Asistente de seguridad de la información · ISO 27001 / ISO 27002")

if not st.session_state.mensajes:
    st.markdown(
        "<div class='empty-state'>"
        "<div class='icon'>🔒</div>"
        "<strong>¿En qué puedo ayudarte?</strong>"
        "Consulta normas, controles o análisis de cumplimiento."
        "<ul>"
        "<li>¿Qué establece el control 8.8 sobre vulnerabilidades técnicas?</li>"
        "<li>¿Cuáles son los requisitos del Anexo A de la ISO 27001?</li>"
        "<li>¿Cumple nuestra política de acceso con la ISO 27002?</li>"
        "<li>Explícame los controles de gestión de activos</li>"
        "</ul>"
        "</div>",
        unsafe_allow_html=True,
    )

# Renderizar mensajes anteriores desde el estado de sesión
for msg in st.session_state.mensajes:
    with st.chat_message(msg["rol"]):
        st.markdown(msg["contenido"])
        if msg.get("fuente_citada"):
            st.caption(msg["fuente_citada"])

# ── Procesamiento de la pregunta ─────────────────────────────────────────────

if pregunta := st.chat_input("Escribe tu pregunta sobre ISO 27001/27002…"):
    historial_previo = st.session_state.mensajes.copy()
    primera_pregunta = not historial_previo

    # Mostrar el mensaje del usuario inmediatamente
    with st.chat_message("user"):
        st.markdown(pregunta)

    # Procesar y mostrar la respuesta del asistente
    fuente_citada = ""
    respuesta_display = ""
    with st.chat_message("assistant"):
        resp_fija = respuesta_identidad(pregunta)
        if resp_fija:
            respuesta = resp_fija
            respuesta_display = respuesta
            st.markdown(respuesta)
        elif es_inyeccion_prompt(pregunta):
            respuesta = (
                "Esta solicitud parece intentar modificar mi comportamiento o rol. "
                "Solo puedo responder preguntas sobre los documentos disponibles."
            )
            respuesta_display = respuesta
            st.warning(respuesta)
        else:
            intencion = detectar_intencion(pregunta)
            with st.spinner("Buscando en la documentación…"):
                chunks, metas = buscar_contexto(
                    coleccion, pregunta, historial=historial_previo, intencion=intencion,
                )

            if not chunks:
                respuesta = "Esta consulta no está cubierta por los documentos disponibles."
                respuesta_display = respuesta
                st.info(respuesta)
            else:
                placeholder = st.empty()
                respuesta = ""
                for token in generar_respuesta(
                    chunks, metas, pregunta,
                    historial=historial_previo, intencion=intencion,
                ):
                    respuesta += token
                    placeholder.markdown(respuesta + "▌")
                respuesta_display = reemplazar_citas(respuesta, metas)
                placeholder.markdown(respuesta_display)
                metas_citadas = extraer_metas_citadas(respuesta, metas, intencion)
                fuente_citada = resumen_fuentes(metas_citadas)
                if fuente_citada:
                    st.caption(fuente_citada)

    # Guardar en estado de sesión y en disco
    st.session_state.mensajes.append({"rol": "user", "contenido": pregunta})
    st.session_state.mensajes.append({
        "rol": "assistant",
        "contenido": respuesta_display,
        "fuente_citada": fuente_citada,
    })
    guardar_mensajes(st.session_state.conv_id, st.session_state.mensajes)

    # Solo rerun en la primera pregunta para actualizar el título en la sidebar
    if primera_pregunta:
        actualizar_titulo(st.session_state.conv_id, pregunta)
        st.rerun()
