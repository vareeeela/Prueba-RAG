# RAG - Sistema de Búsqueda Semántica con ChromaDB

Sistema de Retrieval-Augmented Generation (RAG). Permite hacer preguntas sobre documentos y obtener respuestas basadas en búsqueda semántica.

## Características

- **Búsqueda semántica** de fragmentos de documentos
- **Base de datos vectorial** con ChromaDB
- **Embeddings multilingües** con sentence-transformers
- **Dos modos de inferencia**: LLM local via Ollama o API de Groq
- **Soporte para múltiples formatos**: PDF, Markdown, DOCX, PPTX, XLSX, TXT (via MarkItDown)
- **Chat interactivo** en terminal

## Requisitos Previos

### Software necesario:

1. **Python 3.8+** - Descargar desde [python.org](https://www.python.org/downloads/)
2. **Ollama** *(solo si usas modo local)* - Descargar desde [ollama.ai](https://ollama.ai)
   - Asegúrate de que esté ejecutándose antes de lanzar el script (`ollama serve`)

## Instalación

### 1. Clonar el repositorio

```bash
git clone <https://github.com/vareeeela/Prueba-RAG>
cd Prueba-RAG
```

### 2. Crear y activar entorno virtual

**En Windows (PowerShell):**
```powershell
python -m venv venv
venv\Scripts\Activate
```

**En macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Instalar dependencias

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Elegir modo de inferencia

El modo se controla con la variable `MODO` en `rag_v2.py`:

```python
MODO = "local"   # modelo local via Ollama
# MODO = "groq"  # API Groq
```

Comenta una línea y descomenta la otra para cambiar de modo.

---

#### Modo local (Ollama)

Descarga el modelo que quieras usar:

```bash
ollama pull llama2         # equilibrio calidad/velocidad
ollama pull llama3.2       # más potente, más lento
```

Actualiza `MODELO_LLM` en `rag_v2.py` con el nombre del modelo descargado.

---

#### Modo API (Groq)

1. Crea una cuenta en [console.groq.com](https://console.groq.com) y genera una API key.
2. Crea un archivo `.env` en la raíz del proyecto:

```
GROQ_API_KEY=tu_clave_aqui
```

3. Instala el paquete de Groq:

```bash
pip install groq
```

Modelos recomendados (variable `MODELO_GROQ` en `rag_v2.py`):

| Modelo | Contexto | Cuándo usarlo |
|---|---|---|
| `llama-3.3-70b-versatile` | 128k | Mejor calidad general (por defecto) |
| `mixtral-8x7b-32768` | 32k | Buena velocidad con calidad alta |
| `llama-3.1-8b-instant` | 128k | Respuestas muy rápidas |

---

#### Modelo de embeddings

Independiente del modo LLM, configurable con `MODELO_EMBEDDINGS`:

- `all-MiniLM-L6-v2` — rápido (por defecto)
- `all-MiniLM-L12-v2` — mayor calidad
- `paraphrase-multilingual-MiniLM-L12-v2` — mejor cobertura multilingüe

### 5. Preparar documentos

```bash
# Crear carpeta documentos (si no existe)
mkdir documentos

# Copiar tus archivos PDF, DOCX, Markdown, etc. aquí
# Ejemplo: documentos/documento1.pdf
```

## Uso del Proyecto

### Chat Interactivo (rag_v2.py)

```bash
python rag_v2.py
```

Este script:
1. Lee documentos desde la carpeta `documentos/`
2. Genera embeddings y los almacena en ChromaDB
3. Abre un chat interactivo donde puedes hacer preguntas
4. Escribe `salir` para terminar

## Estructura

```
Prueba-RAG/
├── README.md                  # Documentación del proyecto
├── requirements.txt           # Dependencias Python
├── .gitignore                 # Archivos ignorados en Git
├── .env                       # API keys (no se sube a Git)
├── rag_v2.py                  # Script principal
├── documentos/                # Documentos (PDF, DOCX, etc.)
│   ├── documento1.pdf
│   └── documento2.docx
├── bd_vectorial/              # Base de datos ChromaDB (generada automáticamente)
│   ├── chroma.sqlite3
│   └── [metadatos]
└── venv/                      # Entorno virtual Python
```

## Flujo de Trabajo 

```
1. Coloca tus documentos en documentos/
   ↓
2. Ejecuta: python rag_v2.py
   ↓
3. El script convierte documentos → fragmentos → embeddings
   ↓
4. ChromaDB almacena los embeddings en bd_vectorial/
   ↓
5. Chat: Escribe preguntas y obtén respuestas basadas en tus documentos
   ↓
6. Escribe "salir" para terminar
```

## Recursos 

- [ChromaDB Documentation](https://docs.trychroma.com/)
- [Ollama Models](https://ollama.ai)
- [Groq Console](https://console.groq.com)
- [Sentence Transformers](https://www.sbert.net/)
