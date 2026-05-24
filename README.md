# RAG - Sistema de Búsqueda Semántica con ChromaDB

Sistema de Retrieval-Augmented Generation (RAG). Permite hacer preguntas sobre documentos y obtener respuestas basadas en búsqueda semántica.

## Características

- **Búsqueda semántica** de fragmentos de documentos
- **Base de datos vectorial** con ChromaDB
- **Embeddings multilingües** con sentence-transformers (`paraphrase-multilingual-MiniLM-L12-v2`)
- **Dos modos de inferencia**: LLM local via Ollama o API de Groq
- **Soporte para múltiples formatos**: PDF, Markdown, DOCX, PPTX, XLSX, TXT (via MarkItDown)
- **Configuración centralizada** en `config.yaml`, sin tocar código
- **Chat interactivo** en terminal

## Requisitos Previos

### Software necesario:

1. **Python 3.11+** - Descargar desde [python.org](https://www.python.org/downloads/)
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

El modo se controla en `config.yaml`:

```yaml
llm:
  modo: "local"   # modelo local via Ollama
  # modo: "groq"  # API Groq
```

---

#### Modo local (Ollama)

Descarga el modelo configurado en `config.yaml`:

```bash
ollama pull llama3.2
```

Para cambiar de modelo, edita `config.yaml`:

```yaml
llm:
  modelo_local: "llama3.2"   # o llama3.1:8b, qwen2.5:7b, etc.
```

---

#### Modo API (Groq)

1. Crea una cuenta en [console.groq.com](https://console.groq.com) y genera una API key.
2. Crea un archivo `.env` en la raíz del proyecto:

```
GROQ_API_KEY=tu_clave_aqui
```

Para cambiar de modelo, edita `config.yaml`:

```yaml
llm:
  modelo_groq: "llama-3.3-70b-versatile"
```

Modelos recomendados:

| Modelo | Contexto | Cuándo usarlo |
|---|---|---|
| `llama-3.3-70b-versatile` | 128k | Mejor calidad general (por defecto) |
| `mixtral-8x7b-32768` | 32k | Buena velocidad con calidad alta |
| `llama-3.1-8b-instant` | 128k | Respuestas muy rápidas |

---

#### Modelo de embeddings

Configurable en `config.yaml`:

```yaml
embeddings:
  modelo: "paraphrase-multilingual-MiniLM-L12-v2"  # multilingüe (por defecto)
```

Opciones:

| Modelo | Idiomas | Notas |
|---|---|---|
| `paraphrase-multilingual-MiniLM-L12-v2` | Multilingüe | Por defecto; equilibrio calidad/velocidad |
| `intfloat/multilingual-e5-base` | Multilingüe | Mayor rendimiento; requiere prefijos `query:`/`passage:` |
| `all-MiniLM-L6-v2` | Inglés | Más rápido; no recomendado para español |

> Al cambiar el modelo de embeddings, la base de datos se reindexará automáticamente en la siguiente ejecución.

### 5. Preparar documentos

```bash
mkdir documentos
# Copia tus archivos PDF, DOCX, Markdown, etc. aquí
```

## Uso del Proyecto

```bash
python rag_v2.py
```

El sistema:
1. Lee documentos desde `documentos/`
2. Indexa solo los nuevos o modificados (caché por hash MD5)
3. Abre un chat interactivo para hacer preguntas
4. Escribe `salir` para terminar

## Estructura

```
tfg-rag/
├── README.md
├── requirements.txt
├── config.yaml                # Parámetros de configuración
├── .env                       # API keys (no se sube a Git)
├── .gitignore
├── rag_v2.py                  # Punto de entrada
├── src/
│   ├── config.py              # Carga config.yaml, expone constantes y singletons
│   ├── indexer.py             # Carga, chunking, hashing e indexación
│   ├── retriever.py           # Búsqueda semántica en ChromaDB
│   ├── generator.py           # Construcción del prompt y llamada al LLM
│   └── main.py                # Bucle de conversación
├── documentos/                # Documentos a indexar (PDF, DOCX, etc.)
└── bd_vectorial/              # Base de datos ChromaDB (generada automáticamente)
```

## Flujo de Trabajo

```
1. Coloca tus documentos en documentos/

2. Ejecuta: python rag_v2.py

3. El sistema convierte documentos → fragmentos → embeddings

4. ChromaDB almacena los embeddings en bd_vectorial/

5. Chat: Escribe preguntas y obtén respuestas basadas en tus documentos

6. Escribe "salir" para terminar
```

## Recursos

- [ChromaDB Documentation](https://docs.trychroma.com/)
- [Ollama Models](https://ollama.ai)
- [Groq Console](https://console.groq.com)
- [Sentence Transformers](https://www.sbert.net/)
