"""
ai_service.py - All LLM interactions: question answering, summarization, comparison.
"""

from typing import List, Dict, Optional
from google import genai
from google.genai import types

from app.core.config import settings
from app.core.logging import logger


class AIService:

    def __init__(self):
        self.client = genai.Client(api_key=settings.GEMINI_API_KEY)

    def _call_gemini(self, system_prompt: str, user_prompt: str, temperature: float = 0.2) -> str:
        response = self.client.models.generate_content(
            model=settings.GEMINI_MODEL,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=temperature,
                max_output_tokens=2048,
            )
        )
        return response.text

    # ── Question Answering ────────────────────────────────────────────────────

    def answer_question(
        self,
        question: str,
        context_chunks: List[Dict],
        conversation_history: List[Dict] = None
    ) -> Dict:
        """
        Generate a grounded answer with citations using RAG context.
        """
        if not context_chunks:
            return {
                "answer": "I could not find sufficient information in the uploaded documents to answer this question.",
                "confidence_score": 0.0
            }

        context_text = ""
        for i, chunk in enumerate(context_chunks):
            context_text += (
                f"[Chunk {i+1}] Document: {chunk.get('doc_name', chunk['doc_id'])} | "
                f"Page: {chunk['page_number']}\n{chunk['text']}\n\n"
            )

        history_text = ""
        if conversation_history:
            history_text = "Previous conversation:\n"
            for turn in conversation_history[-6:]:
                history_text += f"{turn['role'].upper()}: {turn['content']}\n"
            history_text += "\n"

        system_prompt = (
            "You are a precise AI Research Assistant. Answer questions ONLY based on "
            "the provided document context. Never fabricate information. "
            "If the context does not contain the answer, say so clearly. "
            "Always cite the document name and page number for each piece of information you use."
        )

        user_prompt = (
            f"{history_text}"
            f"DOCUMENT CONTEXT:\n{context_text}\n"
            f"QUESTION: {question}\n\n"
            "Provide a comprehensive answer based on the context above. "
            "Mention which document and page number supports each point."
        )

        answer = self._call_gemini(system_prompt, user_prompt, temperature=0.1)

        # Confidence based on top chunk score
        top_score = max(c.get("score", 0) for c in context_chunks)
        confidence = min(round(float(top_score), 3), 1.0)

        logger.info(f"QA completed. Confidence: {confidence}")
        return {"answer": answer, "confidence_score": confidence}

    # ── Summarization ─────────────────────────────────────────────────────────

    def summarize(self, doc_name: str, full_text: str, summary_type: str = "executive") -> str:
        """
        Generate different types of summaries for a document.
        """
        # Truncate to avoid token limits
        truncated = full_text[:12000]

        type_instructions = {
            "executive": (
                "Write a concise executive summary (150-200 words) covering: "
                "main objective, key findings, and business/research implications."
            ),
            "technical": (
                "Write a detailed technical summary covering: methodology, technical approach, "
                "algorithms/models used, datasets, experimental setup, and technical results."
            ),
            "bullet": (
                "Summarize the document as a structured bullet point list. "
                "Use main bullets for major sections and sub-bullets for key details. "
                "Aim for 15-25 bullet points total."
            ),
            "key_takeaways": (
                "Extract exactly 5-7 key takeaways from this document. "
                "Each takeaway should be a single, actionable or insightful sentence. "
                "Number them 1 through 7."
            ),
        }

        instruction = type_instructions.get(summary_type, type_instructions["executive"])

        system_prompt = (
            f"You are an expert research analyst. {instruction} "
            "Base your summary strictly on the provided document content."
        )

        user_prompt = f"Document: {doc_name}\n\nContent:\n{truncated}"

        return self._call_gemini(system_prompt, user_prompt, temperature=0.3)

    # ── Comparison ────────────────────────────────────────────────────────────

    def compare_documents(
        self,
        doc_names: List[str],
        contexts: List[Dict],
        aspect: str = "general"
    ) -> str:
        """
        Compare two or more documents on a specific aspect.
        """
        context_sections = ""
        for i, (name, ctx) in enumerate(zip(doc_names, contexts)):
            text = "\n".join(c["text"] for c in ctx[:5])[:3000]
            context_sections += f"DOCUMENT {i+1}: {name}\n{text}\n\n{'─'*60}\n\n"

        aspect_instructions = {
            "general": "Compare these documents across all major dimensions: objectives, methods, findings, strengths, and limitations.",
            "methodology": "Focus the comparison specifically on the research methodologies, experimental designs, and technical approaches.",
            "advantages": "Compare the advantages, strengths, and positive contributions of each document.",
            "differences": "Identify and explain the key differences between these documents.",
            "similarities": "Identify and explain the key similarities and common themes across these documents.",
            "conclusions": "Compare the conclusions, findings, and recommendations from each document.",
        }

        instruction = aspect_instructions.get(aspect, aspect_instructions["general"])

        system_prompt = (
            "You are an expert research analyst skilled at comparative document analysis. "
            f"{instruction} "
            "Structure your comparison clearly with sections for each document and a final synthesis. "
            "Use a markdown table where appropriate to highlight key differences side by side."
        )

        user_prompt = (
            f"Please compare the following {len(doc_names)} documents:\n\n"
            f"{context_sections}"
            f"Comparison aspect: {aspect}"
        )

        return self._call_gemini(system_prompt, user_prompt, temperature=0.3)


# Singleton
ai_service = AIService()
