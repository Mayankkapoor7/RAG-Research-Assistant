import re
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path

from langchain_community.document_loaders import PyMuPDFLoader, TextLoader, WebBaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

_ARXIV_ID_RE = re.compile(r"(\d{4}\.\d{4,5}(?:v\d+)?)")

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    add_start_index=True,
)
_md_splitter = RecursiveCharacterTextSplitter.from_language(
    "markdown",
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    add_start_index=True,
)


def _stamp_metadata(docs: list[Document], file_name: str, title: str) -> list[Document]:
    for doc in docs:
        doc.metadata["file_name"] = file_name
        doc.metadata["source"] = file_name
        doc.metadata["title"] = title
    return docs


def load_pdf(file_path: str, original_filename: str | None = None) -> list[Document]:
    raw_docs = PyMuPDFLoader(file_path).load()
    name = original_filename or Path(file_path).name
    stem = Path(name).stem

    # Preserve 1-indexed page number and filename
    for doc in raw_docs:
        page_idx = doc.metadata.get("page", 0)
        doc.metadata["page"] = int(page_idx) + 1 if isinstance(page_idx, (int, float)) else 1
        doc.metadata["file_name"] = name
        doc.metadata["source"] = name
        doc.metadata["title"] = stem

    split_chunks = _splitter.split_documents(raw_docs)
    for chunk in split_chunks:
        chunk.metadata["file_name"] = name
        chunk.metadata["source"] = name
        chunk.metadata["title"] = stem

    return split_chunks


def load_text(file_path: str, original_filename: str | None = None) -> list[Document]:
    docs = TextLoader(file_path, encoding="utf-8").load()
    name = original_filename or Path(file_path).name
    stem = Path(name).stem
    for doc in docs:
        doc.metadata["page"] = 1
    chunks = _splitter.split_documents(docs)
    return _stamp_metadata(chunks, name, stem)


def load_markdown(file_path: str, original_filename: str | None = None) -> list[Document]:
    docs = TextLoader(file_path, encoding="utf-8").load()
    name = original_filename or Path(file_path).name
    stem = Path(name).stem
    for doc in docs:
        doc.metadata["page"] = 1
    chunks = _md_splitter.split_documents(docs)
    return _stamp_metadata(chunks, name, stem)


def load_webpage(url: str) -> list[Document]:
    docs = WebBaseLoader(url, requests_kwargs={"timeout": 30}).load()
    title = (docs[0].metadata.get("title") or url) if docs else url
    chunks = _splitter.split_documents(docs)
    return _stamp_metadata(chunks, url, title)


def _extract_arxiv_id(query: str) -> str | None:
    m = _ARXIV_ID_RE.search(query)
    if m:
        return re.sub(r"v\d+$", "", m.group(1))
    return None


def _arxiv_api_lookup(arxiv_id: str) -> str:
    url = f"https://export.arxiv.org/api/query?id_list={arxiv_id}"
    with urllib.request.urlopen(url, timeout=10) as resp:
        xml = resp.read().decode()
    titles = re.findall(r"<title>(.*?)</title>", xml, re.DOTALL)
    return titles[1].strip() if len(titles) > 1 else arxiv_id


def _arxiv_search(query: str) -> str:
    phrase = query.strip('"')
    search_query = urllib.parse.quote(f'ti:"{phrase}"')
    url = f"https://export.arxiv.org/api/query?search_query={search_query}&max_results=1&sortBy=relevance"
    with urllib.request.urlopen(url, timeout=15) as resp:
        xml = resp.read().decode()
    m = re.search(r"<id>https?://arxiv\.org/abs/(\d{4}\.\d{4,5}(?:v\d+)?)</id>", xml)
    if not m:
        raise ValueError(f"No ArXiv paper found for: {query}")
    return re.sub(r"v\d+$", "", m.group(1))


def _load_arxiv_by_id(arxiv_id: str) -> list[Document]:
    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}"
    with urllib.request.urlopen(pdf_url, timeout=60) as resp:
        pdf_bytes = resp.read()
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(pdf_bytes)
            tmp_path = tmp.name
        docs = PyMuPDFLoader(tmp_path).load()
        if not docs:
            raise ValueError(f"Could not load PDF for ArXiv ID: {arxiv_id}")
        title = (docs[0].metadata.get("title") or "").strip() or _arxiv_api_lookup(arxiv_id)
        name = f"arxiv_{arxiv_id}.pdf"
        for doc in docs:
            page_idx = doc.metadata.get("page", 0)
            doc.metadata["page"] = int(page_idx) + 1 if isinstance(page_idx, (int, float)) else 1
            doc.metadata["file_name"] = name
            doc.metadata["source"] = name
            doc.metadata["title"] = title
        chunks = _splitter.split_documents(docs)
        return _stamp_metadata(chunks, name, title)
    finally:
        if tmp_path:
            Path(tmp_path).unlink(missing_ok=True)


def load_arxiv(query: str) -> list[Document]:
    arxiv_id = _extract_arxiv_id(query) or _arxiv_search(query)
    return _load_arxiv_by_id(arxiv_id)


def load_document(source: str, original_filename: str | None = None) -> list[Document]:
    """Dispatch to the appropriate loader based on URL prefix or file extension."""
    if source.startswith(("http://", "https://")):
        return load_webpage(source)
    target_name = original_filename or source
    ext = Path(target_name).suffix.lower()
    if ext == ".pdf":
        return load_pdf(source, original_filename=target_name)
    if ext == ".txt":
        return load_text(source, original_filename=target_name)
    if ext in (".md", ".markdown"):
        return load_markdown(source, original_filename=target_name)
    raise ValueError(f"Unsupported file type: {ext}")
