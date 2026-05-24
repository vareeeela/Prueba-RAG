import os

import yaml
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from markitdown import MarkItDown
from rich.console import Console

load_dotenv()

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

with open(os.path.join(_ROOT, "config.yaml")) as _f:
    _cfg = yaml.safe_load(_f)

# Rutas
CARPETA_DOCS = os.path.join(_ROOT, _cfg["rutas"]["documentos"])
RUTA_BD = os.path.join(_ROOT, _cfg["rutas"]["bd_vectorial"])
COLECCION = _cfg["rutas"]["coleccion"]
CACHE_HASHES = os.path.join(RUTA_BD, ".doc_hashes.json")

# Embeddings
MODELO_EMBEDDINGS = _cfg["embeddings"]["modelo"]

# Retrieval
N_RESULTADOS = _cfg["retrieval"]["n_resultados"]
CHUNK_SIZE = _cfg["retrieval"]["chunk_size"]
CHUNK_OVERLAP = _cfg["retrieval"]["chunk_overlap"]
MIN_CHUNK_LEN = _cfg["retrieval"]["min_chunk_len"]
EXTENSIONES = tuple(_cfg["retrieval"]["extensiones"])

# LLM
MODO = _cfg["llm"]["modo"]
MODELO_LLM = _cfg["llm"]["modelo_local"]
MODELO_GROQ = _cfg["llm"]["modelo_groq"]
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

# Singletons compartidos
console = Console()
md_converter = MarkItDown()
splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
embedding_fn = SentenceTransformerEmbeddingFunction(model_name=MODELO_EMBEDDINGS)
