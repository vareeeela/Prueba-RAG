# RAG - Sistema de Búsqueda Semántica con ChromaDB y Ollama

Sistema de Retrieval-Augmented Generation (RAG). Permite hacer preguntas sobre documentos PDF y obtener respuestas basadas en búsqueda semántica.

## Características

- **Búsqueda semántica** de fragmentos de documentos
- **Base de datos vectorial** con ChromaDB
- **Embeddings multilingües** con sentence-transformers
- **Generación de respuestas** con Ollama (LLM local)
- **Soporte para múltiples formatos**: PDF, Markdown, DOCX, etc. (via MarkItDown)
- **Chat interactivo** en terminal

## Requisitos Previos

### Software necesario:

1. **Python 3.8+** - Descargar desde [python.org](https://www.python.org/downloads/)
2. **Ollama** - Descargar desde [ollama.ai](https://ollama.ai)
   - Necesario para ejecutar modelos LLM localmente
   - Asegúrate de que Ollama esté ejecutándose (`ollama serve`)

### Hardware recomendado:

- RAM: Mínimo 8GB (recomendado 16GB+)
- GPU: Opcional pero recomendada para mejor rendimiento
- Espacio disco: ~20GB para modelos de Ollama

## Instalación

### 1️⃣ Clonar el repositorio

```bash
git clone <tu-repo-url>
cd tfg-rag
```

### 2️⃣ Crear y activar entorno virtual

**En Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**En macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3️⃣ Instalar dependencias

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4️⃣ Descargar modelo de Ollama

```bash
ollama pull llama3.2
```

> **Nota**: Este comando descarga el modelo (~5GB). Si prefieres un modelo más ligero, usa `ollama pull neural-chat` (~4GB) y modifica la línea del modelo en los scripts.

### 5️⃣ Preparar documentos

```bash
# Crear carpeta documentos (si no existe)
mkdir documentos

# Copiar tus archivos PDF, DOCX, Markdown, etc. aquí
# Ejemplo: documentos/documento1.pdf
```

## Uso del Proyecto

### Opción 1: Chat Interactivo (rag_test.py)

```bash
python rag_test.py
```

Este script:
1. Lee documentos desde la carpeta `documentos/`
2. Genera embeddings y los almacena en ChromaDB
3. Abre un chat interactivo donde puedes hacer preguntas
4. Escribe `salir` para terminar

### Opción 2: Procesamiento Avanzado (rag_v2.py)

```bash
python rag_v2.py
```

Este script utiliza MarkItDown para mejor conversión de documentos y metadatos más detallados.

### Opción 3: Probar Embeddings (test_embeddings.py)

```bash
python test_embeddings.py
```

Verifica que los embeddings se generan correctamente (útil para diagnosticar problemas).

## Estructura

```
tfg-rag/
├── README.md                 
├── requirements.txt          # Dependencias Python
├── .gitignore              # Archivos ignorados en Git
├── rag_test.py            # Script principal - Chat RAG simple
├── rag_v2.py              # Script avanzado - RAG con MarkItDown
├── test_embeddings.py     # Test de embeddings
├── documentos/            # Documentos (PDF, DOCX, etc.)
│   ├── documento1.pdf
│   └── documento2.docx
├── bd_vectorial/          # Base de datos ChromaDB (generada auto)
│   ├── chroma.sqlite3
│   └── [metadatos]
└── venv/                  # Entorno virtual Python
```

## Flujo de Trabajo 

```
1. Coloca tus documentos en documentos/
   ↓
2. Ejecuta: python rag_test.py (o rag_v2.py)
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
- [Sentence Transformers](https://www.sbert.net/)
