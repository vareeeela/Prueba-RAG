"""
RAG pipeline: índice persistente + embeddings multilingües + respuesta en streaming.
"""
import hashlib
import json
import os

import chromadb
import ollama
from dotenv import load_dotenv
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

load_dotenv()
from langchain_text_splitters import RecursiveCharacterTextSplitter
from markitdown import MarkItDown
from rich.console import Console
from rich.panel import Panel

console = Console()

# ── Configuración ─────────────────────────────────────────────
CARPETA_DOCS = "documentos"
RUTA_BD = "bd_vectorial"
COLECCION = "documentos_rag"
# MODELO_EMBEDDINGS = "paraphrase-multilingual-MiniLM-L12-v2"
MODELO_EMBEDDINGS = "all-MiniLM-L6-v2"

# ── Modo de inferencia: comenta una línea, descomenta la otra ──
MODO = "local"   # modelo local via Ollama
# MODO = "groq"  # API Groq  (requiere GROQ_API_KEY en el entorno)

# ── Config modelo local (Ollama) ───────────────────────────────
# MODELO_LLM = "llama3.2"
MODELO_LLM = "llama2"

# ── Config Groq ────────────────────────────────────────────────
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")   # o pon tu clave directamente
MODELO_GROQ = "llama-3.3-70b-versatile"             # otros: mixtral-8x7b-32768, gemma2-9b-it
N_RESULTADOS = 5 # cantidad de fragmentos a recuperar para contexto
CACHE_HASHES = os.path.join(RUTA_BD, ".doc_hashes.json") 
EXTENSIONES = (".pdf", ".docx", ".pptx", ".xlsx", ".txt", ".md")

# ── Inicialización única ───────────────────────────────────────
md_converter = MarkItDown() 
splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
embedding_fn = SentenceTransformerEmbeddingFunction(model_name=MODELO_EMBEDDINGS)


# ── Helpers internos ───────────────────────────────────────────

def _hash_archivo(ruta: str) -> str:
    h = hashlib.md5()
    with open(ruta, "rb") as f:
        for bloque in iter(lambda: f.read(65536), b""):
            h.update(bloque)
    return h.hexdigest()


def _cargar_hashes() -> dict:
    if os.path.exists(CACHE_HASHES):
        with open(CACHE_HASHES) as f:
            return json.load(f)
    return {}


def _guardar_hashes(hashes: dict) -> None:
    os.makedirs(RUTA_BD, exist_ok=True)
    with open(CACHE_HASHES, "w") as f:
        json.dump(hashes, f)


def _obtener_coleccion(cliente: chromadb.PersistentClient) -> chromadb.Collection:
    """Obtiene o crea la colección. Si el modelo de embeddings cambió, la recrea."""
    nombres = [c.name for c in cliente.list_collections()]
    if COLECCION in nombres:
        col = cliente.get_collection(name=COLECCION, embedding_function=embedding_fn)
        if col.metadata.get("embedding_model") != MODELO_EMBEDDINGS:
            console.print("[yellow]Modelo de embeddings actualizado — reindexando...[/yellow]")
            cliente.delete_collection(COLECCION)
            if os.path.exists(CACHE_HASHES):
                os.remove(CACHE_HASHES)
        else:
            return col
    return cliente.create_collection(
        name=COLECCION,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine", "embedding_model": MODELO_EMBEDDINGS},
    )


# ── Indexación ─────────────────────────────────────────────────

def indexar_documentos(coleccion: chromadb.Collection) -> None:
    """Indexa solo los documentos nuevos o modificados (detección por hash MD5)."""
    archivos = [
        f for f in os.listdir(CARPETA_DOCS)
        if f.lower().endswith(EXTENSIONES)
    ]

    if not archivos:
        console.print("[red]No hay documentos en 'documentos/'[/red]")
        return

    hashes_previos = _cargar_hashes()
    hashes_nuevos = {}
    cambios = False

    for archivo in archivos:
        ruta = os.path.join(CARPETA_DOCS, archivo)
        hash_actual = _hash_archivo(ruta)
        hashes_nuevos[archivo] = hash_actual

        if hashes_previos.get(archivo) == hash_actual:
            console.print(f"   [dim]Sin cambios: {archivo}[/dim]")
            continue

        console.print(f"   [cyan]Indexando: {archivo}[/cyan]")
        try:
            texto = md_converter.convert(ruta).text_content
            chunks = [c for c in splitter.split_text(texto) if len(c.strip()) > 50]

            # Eliminar versión anterior del mismo documento si existe
            try:
                existentes = coleccion.get(where={"fuente": archivo})
                if existentes["ids"]:
                    coleccion.delete(ids=existentes["ids"])
            except Exception:
                pass

            coleccion.add(
                documents=chunks,
                metadatas=[{"fuente": archivo}] * len(chunks),
                ids=[f"{archivo}_{i}" for i in range(len(chunks))],
            )
            cambios = True
            console.print(f"      [green]→ {len(chunks)} fragmentos[/green]")

        except Exception as e:
            console.print(f"      [red]Error procesando {archivo}: {e}[/red]")

    if cambios or hashes_nuevos != hashes_previos:
        _guardar_hashes(hashes_nuevos)


# ── Recuperación y respuesta ───────────────────────────────────

def _buscar_contexto(coleccion: chromadb.Collection, pregunta: str):
    res = coleccion.query(
        query_texts=[pregunta],
        n_results=N_RESULTADOS,
        include=["documents", "metadatas", "distances"],
    )
    docs = res["documents"][0]
    fuentes = [m["fuente"] for m in res["metadatas"][0]]
    return docs, fuentes


def preguntar(coleccion: chromadb.Collection, pregunta: str) -> None:
    chunks, fuentes = _buscar_contexto(coleccion, pregunta)
    docs_unicos = sorted(set(fuentes))

    contexto = "\n---\n".join(
        f"[Fragmento {i + 1} · {fuente}]\n{chunk}"
        for i, (chunk, fuente) in enumerate(zip(chunks, fuentes))
    )

    prompt = f"""Eres un asistente experto. Responde ÚNICAMENTE con la información del contexto proporcionado.
Cita los documentos de los que extraes la información.
Si la respuesta no está en el contexto, di exactamente:
"No dispongo de información suficiente en la documentación para responder esta consulta."

CONTEXTO (fuentes: {', '.join(docs_unicos)}):
{contexto}

PREGUNTA: {pregunta}

RESPUESTA:"""

    console.print("\n[bold]Respuesta:[/bold] ", end="")

    if MODO == "local":
        # ── Ollama (local) ─────────────────────────────────────
        stream = ollama.chat(
            model=MODELO_LLM,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
        )
        for parte in stream:
            print(parte["message"]["content"], end="", flush=True)

    else:
        # ── Groq (API) ─────────────────────────────────────────
        # Requiere: pip install groq
        from groq import Groq
        cliente_groq = Groq(api_key=GROQ_API_KEY)
        stream = cliente_groq.chat.completions.create(
            model=MODELO_GROQ,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
        )
        for chunk in stream:
            print(chunk.choices[0].delta.content or "", end="", flush=True)

    print()
    console.print(f"\n[dim]Fuentes: {', '.join(docs_unicos)}[/dim]")
    console.print("[dim]" + "─" * 60 + "[/dim]")


# ── Main ───────────────────────────────────────────────────────

def main() -> None:
    console.print(Panel("[bold cyan]Sistema RAG[/bold cyan]", expand=False))

    cliente = chromadb.PersistentClient(path=RUTA_BD)
    coleccion = _obtener_coleccion(cliente)

    console.print("\n[bold]Verificando documentos...[/bold]")
    indexar_documentos(coleccion)
    console.print(f"\n[green]Base de conocimiento lista: {coleccion.count()} fragmentos[/green]\n")
    console.print("Escribe [bold]'salir'[/bold] para terminar.\n" + "─" * 60)

    while True:
        try:
            pregunta = input("\nPregunta: ").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[yellow]Saliendo...[/yellow]")
            break

        if pregunta.lower() == "salir":
            console.print("[yellow]Saliendo...[/yellow]")
            break

        if not pregunta:
            continue

        console.print("[dim]Buscando en documentación...[/dim]")
        preguntar(coleccion, pregunta)


if __name__ == "__main__":
    main()
