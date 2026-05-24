from markitdown import MarkItDown
import chromadb
import ollama
import os

# Modelo multilingüe mejorado para búsqueda semántica (mejor soporte para español)
# Inicializar MarkItDown
md_converter = MarkItDown()



# Convertir solo el primer PDF y ver qué sale
ruta = "documentos/documento3.pdf"
resultado = md_converter.convert(ruta)
texto = resultado.text_content

print("PRIMEROS 1000 CARACTERES DEL TEXTO CONVERTIDO:")
print("=" * 60)
print(texto[:1000])
print("=" * 60)
print(f"\nTiene títulos con #: {'#' in texto}")
print(f"Total de caracteres: {len(texto)}")

# ============================================================
# PASO 1: CONVERTIR Y LEER DOCUMENTOS
# ============================================================

def leer_documento(ruta):
    """
    Convierte cualquier formato a Markdown y devuelve el texto.
    Soporta PDF, DOCX, PPTX, Excel, imágenes...
    """
    resultado = md_converter.convert(ruta)
    return resultado.text_content

def trocear_texto(texto, tamanio=1000, solapamiento=150):
    chunks = []
    inicio = 0
    while inicio < len(texto):
        fin = inicio + tamanio
        chunk = texto[inicio:fin]
        if chunk.strip() and len(chunk) > 50:
            chunks.append(chunk)
        inicio += tamanio - solapamiento
    return chunks

def cargar_documentos(carpeta):
    """
    Lee todos los documentos de la carpeta, los convierte a Markdown
    y los trocea de forma inteligente.
    """
    extensiones = (".pdf", ".docx", ".pptx", ".xlsx", ".txt", ".md")
    archivos = [f for f in os.listdir(carpeta) if f.lower().endswith(extensiones)]

    if not archivos:
        print("❌ No hay documentos en la carpeta 'documentos'")
        return [], [], []

    todos_chunks = []
    todos_metadatos = []
    todos_ids = []
    contador = 0

    for archivo in archivos:
        ruta = os.path.join(carpeta, archivo)
        print(f"   📄 Convirtiendo: {archivo}")

        try:
            texto = leer_documento(ruta)
            chunks = trocear_texto(texto)

            for chunk in chunks:
                todos_chunks.append(chunk)
                todos_metadatos.append({"fuente": archivo})
                todos_ids.append(f"chunk_{contador}")
                contador += 1

            print(f"      → {len(chunks)} fragmentos extraídos")

        except Exception as e:
            print(f"      ⚠️ Error procesando {archivo}: {e}")
            continue

    return todos_chunks, todos_metadatos, todos_ids

print("📚 Cargando documentos...")
chunks, metadatos, ids = cargar_documentos("documentos")
print(f"\n   → Total: {len(chunks)} fragmentos de {len(set(m['fuente'] for m in metadatos))} documentos")

# ============================================================
# PASO 2: GUARDAR EN CHROMADB CON METADATOS
# ============================================================

print("\n🧠 Guardando en base vectorial...")

cliente = chromadb.PersistentClient(path="bd_vectorial")

try:
    cliente.delete_collection("documentos_sgsi")
except:
    pass

coleccion = cliente.create_collection(
    name="documentos_sgsi",
    metadata={"hnsw:space": "cosine"}
)


coleccion.add(
    documents=chunks,
    metadatas=metadatos,
    ids=ids
)

print(f"   → Guardados correctamente")

# ============================================================
# PASO 3: BÚSQUEDA CON TRAZABILIDAD
# ============================================================

def buscar_chunks_relevantes(pregunta, n_resultados=5):
    resultados = coleccion.query(
        query_texts=[pregunta],
        n_results=n_resultados,
        include=["documents", "metadatas", "distances"]
    )
    chunks_encontrados = resultados["documents"][0]
    fuentes = [m["fuente"] for m in resultados["metadatas"][0]]
    distancias = resultados["distances"][0]
    return chunks_encontrados, fuentes, distancias

# ============================================================
# PASO 4: GENERAR RESPUESTA CON TRAZABILIDAD
# ============================================================

def preguntar(pregunta):
    chunks_relevantes, fuentes, distancias = buscar_chunks_relevantes(pregunta)
    
    # Construir contexto indicando de qué documento viene cada fragmento
    contexto_partes = []
    for i, (chunk, fuente, distancia) in enumerate(zip(chunks_relevantes, fuentes, distancias)):
        contexto_partes.append(f"[FRAGMENTO {i+1} - Fuente: {fuente}]\n{chunk}\n")
    contexto = "\n---\n".join(contexto_partes)
    
    # Documentos únicos encontrados
    docs_unicos = list(set(fuentes))
    docs_str = ", ".join(docs_unicos)
    
    prompt = f"""Eres un asistente especializado en historia de España.

Basa tu respuesta ÚNICAMENTE en el contexto proporcionado a continuación. 
Responde de forma clara y directa, citando los DOCUMENTOS de los que extraes la información (no fragmentos).
Si NO encuentras la información en el contexto, responde: "No dispongo de información suficiente en la documentación para responder esta consulta."

CONTEXTO (de los documentos: {docs_str}):
{contexto}

PREGUNTA:
{pregunta}

RESPUESTA (cita los documentos de origen, no los fragmentos):"""

    respuesta = ollama.chat(
        model="llama3.2",
        messages=[{"role": "user", "content": prompt}]
    )
    
    return respuesta["message"]["content"], docs_unicos, chunks_relevantes

# DIAGNÓSTICO - quitar cuando funcione
print("\n🔬 TEST DE RECUPERACIÓN:")
chunks_test, fuentes_test, distancias_test = buscar_chunks_relevantes("cuándo murió enrique IV")
for i, (chunk, fuente, distancia) in enumerate(zip(chunks_test, fuentes_test, distancias_test)):
    print(f"\n--- Fragmento {i+1} (distancia: {distancia:.3f}) ---")
    print(f"Fuente: {fuente}")
    print(f"Texto: {chunk[:200]}...")

print("\n🔬 BÚSQUEDA DIRECTA DE 'Enrique':")
todos = coleccion.get(include=["documents", "metadatas"])
for doc, meta in zip(todos["documents"], todos["metadatas"]):
    if "nrique IV" in doc:
        print(f"\nFuente: {meta['fuente']}")
        print(f"Texto: {doc[:300]}")
        print("---")
# ============================================================
# PASO 5: CHAT INTERACTIVO
# ============================================================

print("\n Sistema listo. Escribe 'salir' para terminar.\n")
print("=" * 60)

while True:
    pregunta = input("\n🔍 Tu pregunta: ").strip()
    
    if pregunta.lower() == "salir":
        print("Saliendo...")
        break
    
    if not pregunta:
        continue
    
    print("\n⏳ Buscando en documentación...")
    respuesta, docs_consultados, chunks = preguntar(pregunta)
    
    print(f"\n💬 Respuesta:\n{respuesta}")
    
    # Mostrar documentos únicos consultados
    print(f"\n📎 Documentos consultados: {', '.join(docs_consultados)}")
    print("-" * 60)



