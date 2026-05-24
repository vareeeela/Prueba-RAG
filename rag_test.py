import chromadb
import ollama
import fitz  # PyMuPDF
import os

# ============================================================
# PASO 1: LEER Y TROCEAR EL DOCUMENTO
# ============================================================

def leer_pdf(ruta):
    """Lee un PDF y devuelve el texto completo"""
    doc = fitz.open(ruta)
    texto = ""
    for pagina in doc:
        texto += pagina.get_text()
    return texto

def trocear_texto(texto, tamanio=500, solapamiento=50):
    """
    Divide el texto en fragmentos (chunks).
    - tamanio: caracteres por fragmento
    - solapamiento: caracteres compartidos entre fragmentos consecutivos
      (evita que una idea quede cortada entre dos chunks)
    """
    chunks = []
    inicio = 0
    while inicio < len(texto):
        fin = inicio + tamanio
        chunk = texto[inicio:fin]
        if chunk.strip():  # ignorar chunks vacíos
            chunks.append(chunk)
        inicio += tamanio - solapamiento
    return chunks

print("📄 Leyendo documento...")
texto = leer_pdf("documento1.pdf")
chunks = trocear_texto(texto)
print(f"   → {len(chunks)} fragmentos generados")

# ============================================================
# PASO 2: GENERAR EMBEDDINGS Y GUARDAR EN CHROMADB
# ============================================================

print("🧠 Generando embeddings y guardando en base vectorial...")

# Inicializar ChromaDB en local (crea una carpeta 'bd_vectorial')
cliente = chromadb.PersistentClient(path="bd_vectorial")

# Eliminar colección anterior si existe (útil al re-ejecutar)
try:
    cliente.delete_collection("documentos_sgsi")
except:
    pass

coleccion = cliente.create_collection(
    name="documentos_sgsi",
    metadata={"hnsw:space": "cosine"}  # búsqueda por similitud coseno
)

# Añadir los chunks a la colección
coleccion.add(
    documents=chunks,
    ids=[f"chunk_{i}" for i in range(len(chunks))]
)

print(f"   → {len(chunks)} fragmentos guardados en ChromaDB")

# ============================================================
# PASO 3: FUNCIÓN DE BÚSQUEDA (RETRIEVAL)
# ============================================================

def buscar_chunks_relevantes(pregunta, n_resultados=3):
    """
    Busca los fragmentos más relevantes para una pregunta.
    ChromaDB convierte la pregunta a embedding internamente
    y devuelve los n_resultados más similares.
    """
    resultados = coleccion.query(
        query_texts=[pregunta],
        n_results=n_resultados
    )
    return resultados["documents"][0]  # lista de chunks relevantes

# ============================================================
# PASO 4: GENERAR RESPUESTA CON OLLAMA
# ============================================================

def preguntar(pregunta):
    """Pipeline completo: busca contexto relevante y genera respuesta"""
    
    # Recuperar chunks relevantes
    chunks_relevantes = buscar_chunks_relevantes(pregunta)
    contexto = "\n\n---\n\n".join(chunks_relevantes)
    
    # Construir el prompt
    prompt = f"""Eres un asistente especializado en seguridad de la información y normativa ISO 27001.
Responde ÚNICAMENTE basándote en el siguiente contexto extraído de documentación interna.
Si la respuesta no está en el contexto, di exactamente: "No dispongo de información suficiente en la documentación para responder esta consulta."
No inventes información.

CONTEXTO:
{contexto}

PREGUNTA:
{pregunta}

RESPUESTA:"""

    # Llamar a Ollama
    respuesta = ollama.chat(
        model="llama3.2",
        messages=[{"role": "user", "content": prompt}]
    )
    
    return respuesta["message"]["content"], chunks_relevantes

# ============================================================
# PASO 5: CHAT INTERACTIVO EN LA TERMINAL
# ============================================================

print("\n✅ Sistema RAG listo. Escribe 'salir' para terminar.\n")
print("=" * 60)

while True:
    pregunta = input("\n🔍 Tu pregunta: ").strip()
    
    if pregunta.lower() == "salir":
        print("Saliendo...")
        break
    
    if not pregunta:
        continue
    
    print("\n⏳ Buscando en documentación...")
    respuesta, chunks_usados = preguntar(pregunta)
    
    print(f"\n💬 Respuesta:\n{respuesta}")
    print(f"\n📎 Basado en {len(chunks_usados)} fragmentos del documento")