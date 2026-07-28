"""
rag_pipeline.py - Document processing pipeline: extract → chunk → embed → index.
Also handles semantic search and keyword search across FAISS indexes.
"""

import os
import json
import re
import numpy as np
from typing import List, Dict, Optional

import faiss
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from google import genai

from app.core.config import settings
from app.core.logging import logger


class RAGPipeline:

    def __init__(self):
        self.client = genai.Client(api_key=settings.GEMINI_API_KEY)
        os.makedirs(settings.INDEX_DIR, exist_ok=True)

    # ── Text Extraction ───────────────────────────────────────────────────────

    def extract_text_from_pdf(self, filepath: str) -> List[Dict]:
        """
        Extract text page by page from a PDF.
        Returns list of {"page": int, "text": str}
        """
        reader = PdfReader(filepath)
        pages = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            text = self._clean_text(text)
            if text.strip():
                pages.append({"page": i + 1, "text": text})
        logger.info(f"Extracted {len(pages)} pages from {filepath}")
        return pages

    def _clean_text(self, text: str) -> str:
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\x20-\x7E\n]', '', text)
        return text.strip()

    # ── Chunking ──────────────────────────────────────────────────────────────

    def chunk_pages(self, pages: List[Dict]) -> List[Dict]:
        """
        Split each page's text into overlapping chunks.
        Returns list of {"page": int, "chunk_index": int, "text": str}
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        chunks = []
        global_idx = 0
        for page_data in pages:
            page_chunks = splitter.split_text(page_data["text"])
            for chunk in page_chunks:
                if chunk.strip():
                    chunks.append({
                        "page": page_data["page"],
                        "chunk_index": global_idx,
                        "text": chunk.strip()
                    })
                    global_idx += 1
        logger.info(f"Created {len(chunks)} chunks")
        return chunks

    # ── Embedding ─────────────────────────────────────────────────────────────

    def get_embedding(self, text: str) -> List[float]:
        response = self.client.models.embed_content(
            model=settings.EMBEDDING_MODEL,
            contents=text
        )
        if hasattr(response, 'embeddings'):
            return response.embeddings[0].values
        elif hasattr(response, 'embedding'):
            return response.embedding.values
        raise ValueError("Unexpected embedding response format")

    def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of texts one by one (Gemini free tier has no batch endpoint)."""
        embeddings = []
        for i, text in enumerate(texts):
            emb = self.get_embedding(text)
            embeddings.append(emb)
            if (i + 1) % 10 == 0:
                logger.info(f"  Embedded {i+1}/{len(texts)} chunks...")
        return embeddings

    # ── Indexing ──────────────────────────────────────────────────────────────

    def build_index(self, doc_id: str, chunks: List[Dict]) -> int:
        """
        Embed all chunks and build a FAISS index for a single document.
        Saves index + metadata to disk.
        Returns number of chunks indexed.
        """
        texts = [c["text"] for c in chunks]
        embeddings = self.get_embeddings_batch(texts)

        vectors = np.array(embeddings, dtype="float32")
        faiss.normalize_L2(vectors)

        dim = vectors.shape[1]
        index = faiss.IndexFlatIP(dim)   # Cosine similarity via inner product
        index.add(vectors)

        index_path = os.path.join(settings.INDEX_DIR, f"{doc_id}.index")
        meta_path  = os.path.join(settings.INDEX_DIR, f"{doc_id}.json")

        faiss.write_index(index, index_path)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False)

        logger.info(f"Index built for doc {doc_id}: {len(chunks)} vectors")
        return len(chunks)

    def delete_index(self, doc_id: str):
        for ext in [".index", ".json"]:
            path = os.path.join(settings.INDEX_DIR, f"{doc_id}{ext}")
            if os.path.exists(path):
                os.remove(path)

    # ── Search ────────────────────────────────────────────────────────────────

    def _load_index(self, doc_id: str):
        index_path = os.path.join(settings.INDEX_DIR, f"{doc_id}.index")
        meta_path  = os.path.join(settings.INDEX_DIR, f"{doc_id}.json")
        if not os.path.exists(index_path):
            return None, []
        index = faiss.read_index(index_path)
        with open(meta_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        return index, chunks

    def semantic_search(
        self,
        query: str,
        doc_ids: List[str],
        top_k: int = 5
    ) -> List[Dict]:
        """
        Search across one or more document indexes using cosine similarity.
        Returns ranked list of chunk hits.
        """
        query_vec = np.array([self.get_embedding(query)], dtype="float32")
        faiss.normalize_L2(query_vec)

        all_results = []
        for doc_id in doc_ids:
            index, chunks = self._load_index(doc_id)
            if index is None or index.ntotal == 0:
                continue
            k = min(top_k, index.ntotal)
            scores, indices = index.search(query_vec, k)
            for score, idx in zip(scores[0], indices[0]):
                if idx == -1:
                    continue
                chunk = chunks[idx]
                all_results.append({
                    "doc_id": doc_id,
                    "page_number": chunk["page"],
                    "chunk_index": chunk["chunk_index"],
                    "text": chunk["text"],
                    "score": float(score)
                })

        all_results.sort(key=lambda x: x["score"], reverse=True)
        return all_results[:top_k]

    def keyword_search(
        self,
        query: str,
        doc_ids: List[str],
        top_k: int = 5
    ) -> List[Dict]:
        """
        Simple keyword frequency search as a fallback / alternative mode.
        """
        keywords = set(re.findall(r'\w+', query.lower()))
        all_results = []

        for doc_id in doc_ids:
            _, chunks = self._load_index(doc_id)
            for chunk in chunks:
                text_lower = chunk["text"].lower()
                words = set(re.findall(r'\w+', text_lower))
                matches = len(keywords & words)
                if matches > 0:
                    score = matches / len(keywords)
                    all_results.append({
                        "doc_id": doc_id,
                        "page_number": chunk["page"],
                        "chunk_index": chunk["chunk_index"],
                        "text": chunk["text"],
                        "score": score
                    })

        all_results.sort(key=lambda x: x["score"], reverse=True)
        return all_results[:top_k]

    def hybrid_search(
        self,
        query: str,
        doc_ids: List[str],
        top_k: int = 5
    ) -> List[Dict]:
        """
        Combine semantic and keyword scores (0.7 semantic + 0.3 keyword).
        """
        semantic = self.semantic_search(query, doc_ids, top_k=top_k * 2)
        keyword  = self.keyword_search(query, doc_ids, top_k=top_k * 2)

        scores: Dict[str, dict] = {}
        for r in semantic:
            key = f"{r['doc_id']}_{r['chunk_index']}"
            scores[key] = {**r, "hybrid_score": 0.7 * r["score"]}
        for r in keyword:
            key = f"{r['doc_id']}_{r['chunk_index']}"
            if key in scores:
                scores[key]["hybrid_score"] += 0.3 * r["score"]
            else:
                scores[key] = {**r, "hybrid_score": 0.3 * r["score"]}

        results = sorted(scores.values(), key=lambda x: x["hybrid_score"], reverse=True)
        for r in results:
            r["score"] = r.pop("hybrid_score")
        return results[:top_k]

    def get_full_text(self, doc_id: str) -> str:
        """Return all chunks concatenated for summarization."""
        _, chunks = self._load_index(doc_id)
        return "\n\n".join(c["text"] for c in chunks)


# Singleton
rag_pipeline = RAGPipeline()
