import os

import yaml
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from markitdown import MarkItDown
from rich.console import Console
from sentence_transformers import SentenceTransformer

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
SIMILARITY_THRESHOLD = _cfg["retrieval"]["similarity_threshold"]
N_QUERY_VARIANTS = _cfg["retrieval"]["n_query_variants"]
EXTENSIONES = tuple(_cfg["retrieval"]["extensiones"])

# LLM
MODO = _cfg["llm"]["modo"]
MODELO_LLM = _cfg["llm"]["modelo_local"]
MODELO_GROQ = _cfg["llm"]["modelo_groq"]
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
MAX_TURNOS_HISTORIAL = _cfg["llm"]["max_turnos_historial"]
TEMPERATURE = _cfg["llm"]["temperature"]
MAX_TOKENS = _cfg["llm"]["max_tokens"]
TOP_P = _cfg["llm"]["top_p"]

# Singletons compartidos
console = Console()
md_converter = MarkItDown()
splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

_USE_E5_PREFIX = "e5" in MODELO_EMBEDDINGS.lower()
_st_model = SentenceTransformer(MODELO_EMBEDDINGS)


class _E5EmbeddingFunction:
    """Embedding function con soporte de prefijos query/passage para modelos E5."""

    def name(self) -> str:
        return MODELO_EMBEDDINGS

    def __call__(self, input: list[str]) -> list[list[float]]:
        texts = [f"passage: {t}" for t in input] if _USE_E5_PREFIX else input
        return _st_model.encode(texts, normalize_embeddings=True).tolist()

    def embed_query(self, text: str) -> list[float]:
        text = f"query: {text}" if _USE_E5_PREFIX else text
        return _st_model.encode(text, normalize_embeddings=True).tolist()


embedding_fn = _E5EmbeddingFunction()
