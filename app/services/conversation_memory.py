"""
conversation_memory.py - In-memory session-based conversation history manager.
"""

from datetime import datetime
from typing import List, Dict, Optional
from app.core.logging import logger


class ConversationMemory:
    """
    Maintains per-session conversation history for multi-turn QA.
    Stores last N turns per session to manage context window size.
    """

    MAX_TURNS = 20

    def __init__(self):
        self._sessions: Dict[str, List[Dict]] = {}

    def add_turn(self, session_id: str, role: str, content: str):
        if session_id not in self._sessions:
            self._sessions[session_id] = []
        turn = {
            "role": role,
            "content": content,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
        self._sessions[session_id].append(turn)

        # Keep only last MAX_TURNS
        if len(self._sessions[session_id]) > self.MAX_TURNS:
            self._sessions[session_id] = self._sessions[session_id][-self.MAX_TURNS:]

        logger.debug(f"Session {session_id}: added {role} turn")

    def get_history(self, session_id: str) -> List[Dict]:
        return self._sessions.get(session_id, [])

    def clear_session(self, session_id: str):
        if session_id in self._sessions:
            del self._sessions[session_id]
            logger.info(f"Session cleared: {session_id}")

    def get_context_for_llm(self, session_id: str, last_n: int = 6) -> List[Dict]:
        """Return the last N turns for LLM context injection."""
        history = self.get_history(session_id)
        return history[-last_n:] if len(history) > last_n else history

    def list_sessions(self) -> List[str]:
        return list(self._sessions.keys())


# Singleton
conversation_memory = ConversationMemory()
