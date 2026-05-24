# lucIA — Sistema RAG

Sistema de Retrieval-Augmented Generation (RAG). Permite hacer preguntas sobre documentos y obtener respuestas fundamentadas en búsqueda semántica, con interfaz web y memoria conversacional.

## Características

- **Búsqueda semántica** con embeddings multilingües (`intfloat/multilingual-e5-large`)
- **Multi-query**: genera variantes de cada pregunta para mejorar el retrieval
- **Referencias exactas**: cita la página (PDFs) o sección (otros formatos) del documento fuente
- **Memoria conversacional**: el LLM recuerda los turnos anteriores de la misma sesión
- **Historial persistente**: las conversaciones se guardan entre sesiones
- **Interfaz web** con Streamlit (`app.py`) y **CLI** (`rag_v2.py`)
- **Dos modos de inferencia**: LLM local via Ollama o API de Groq
- **Soporte multi-formato**: PDF, DOCX, PPTX, XLSX, TXT, Markdown
- **Configuración centralizada** en `config.yaml`, sin tocar código
- **Reindexación automática** al añadir documentos o cambiar el modelo de embeddings

## Requisitos previos

- **Python 3.11+**
- **Ollama** *(solo si usas modo local)* — [ollama.ai](https://ollama.ai). Debe estar ejecutándose (`ollama serve`) antes de lanzar el sistema.

## Instalación

### 1. Clonar el repositorio

```bash
git clone https://github.com/vareeeela/Prueba-RAG
cd Prueba-RAG
```

### 2. Crear y activar entorno virtual

**Windows (PowerShell):**
```powershell
python -m venv venv
venv\Scripts\Activate
```

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Instalar dependencias

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configurar el modo de inferencia

Edita `config.yaml`:

```yaml
llm:
  modo: "groq"    # "groq" o "local"
```

---

#### Modo API (Groq)

1. Crea una cuenta en [console.groq.com](https://console.groq.com) y genera una API key.
2. Crea un archivo `.env` en la raíz:

```
GROQ_API_KEY=tu_clave_aqui
```

Modelos disponibles en `config.yaml`:

| Modelo | Contexto | Cuándo usarlo |
|---|---|---|
| `llama-3.3-70b-versatile` | 128k | Mejor calidad general (por defecto) |
| `mixtral-8x7b-32768` | 32k | Buena velocidad con calidad alta |
| `llama-3.1-8b-instant` | 128k | Respuestas muy rápidas |

---

#### Modo local (Ollama)

```bash
ollama pull llama3.2
```

Cambia el modelo en `config.yaml`:

```yaml
llm:
  modelo_local: "llama3.2"
```

---

#### Modelo de embeddings

Configurable en `config.yaml`. Al cambiar el modelo la base de datos se reindexará automáticamente.

| Modelo | Tamaño | Notas |
|---|---|---|
| `intfloat/multilingual-e5-large` | ~560 MB | Por defecto; mejor calidad para español |
| `intfloat/multilingual-e5-base` | ~280 MB | Más ligero; buena alternativa |
| `paraphrase-multilingual-MiniLM-L12-v2` | ~120 MB | Más rápido; menor precisión |

### 5. Añadir documentos

Copia tus archivos en la carpeta `documentos/`:

```bash
mkdir documentos
# Copia aquí tus PDF, DOCX, TXT, etc.
```

## Uso

### Interfaz web (recomendado)

```bash
streamlit run app.py
```

Se abre automáticamente en `http://localhost:8501`.

### CLI

```bash
python rag_v2.py
```

Escribe `salir` para terminar.

---

El sistema en ambos modos:
1. Detecta documentos nuevos o modificados y los reindexará (caché por hash MD5)
2. Genera variantes de la pregunta para mejorar el retrieval
3. Recupera los fragmentos más relevantes con umbral de similitud
4. Responde citando el documento, página o sección exacta

## Estructura

```
tfg-rag/
├── README.md
├── requirements.txt
├── config.yaml                # Parámetros de configuración
├── .env                       # API keys (no se sube a Git)
├── .gitignore
├── rag_v2.py                  # Punto de entrada CLI
├── app.py                     # Interfaz web Streamlit
├── src/
│   ├── config.py              # Carga config.yaml, constantes y singletons
│   ├── indexer.py             # Chunking por página/sección, hashing e indexación
│   ├── retriever.py           # Multi-query y búsqueda semántica en ChromaDB
│   ├── generator.py           # Prompt con historial conversacional y llamada al LLM
│   └── main.py                # Bucle CLI
├── documentos/                # Documentos a indexar (no se suben a Git)
└── bd_vectorial/              # Base de datos ChromaDB e historial (generados automáticamente)
```

## Configuración completa (`config.yaml`)

```yaml
embeddings:
  modelo: "intfloat/multilingual-e5-large"

retrieval:
  n_resultados: 5           # Fragmentos recuperados por variante de pregunta
  chunk_size: 800           # Tamaño máximo de cada fragmento (tokens)
  chunk_overlap: 150        # Solapamiento entre fragmentos consecutivos
  min_chunk_len: 50         # Longitud mínima para indexar un fragmento
  similarity_threshold: 0.5 # Distancia coseno máxima (0=idéntico, 1=sin relación)
  n_query_variants: 3       # Número de reformulaciones de la pregunta

llm:
  modo: "groq"
  modelo_groq: "llama-3.3-70b-versatile"
  modelo_local: "llama3.2"
  max_turnos_historial: 6   # Mensajes previos que ve el LLM (para memoria conversacional)
```

## Recursos

- [ChromaDB Documentation](https://docs.trychroma.com/)
- [Ollama Models](https://ollama.ai)
- [Groq Console](https://console.groq.com)
- [Sentence Transformers](https://www.sbert.net/)
- [Streamlit Documentation](https://docs.streamlit.io/)
