import os
from typing import Any, Dict, List

try:
    from anthropic import Anthropic
except ImportError:  # pragma: no cover
    Anthropic = None  # type: ignore

try:
    import google.generativeai as genai
except ImportError:  # pragma: no cover
    genai = None  # type: ignore

from rag.retriever import format_retrieved_sessions_for_prompt


SYSTEM_PROMPT = """You are NeuroPulse AI's personal fatigue coach. You have access to this user's past fatigue sessions and intervention history. Be specific, data-driven, and personal. Never give generic advice. Always cite the retrieved sessions with dates and recovery percentages. Keep response under 120 words. Format: 2-3 sentences of insight + 1 bold recommendation."""


def _build_user_prompt(current_state: Dict[str, Any], retrieved_sessions: List[Dict[str, Any]]) -> str:
    retrieved_sessions_formatted = format_retrieved_sessions_for_prompt(retrieved_sessions)
    return f"""Current fatigue state:
- Fatigue Level: {current_state['fatigue_level']}
- EAR: {current_state['EAR']} | MAR: {current_state['MAR']} | MCR: {current_state['MCR']}
- MPTI: {current_state['MPTI']} | ITSL: {current_state['ITSL']}ms | TRCV: {current_state['TRCV']} | DCR: {current_state['DCR']}
- Time of day: {current_state['time_of_day']}
- Session type: {current_state['session_type']}

Retrieved from your personal history:
{retrieved_sessions_formatted}

Based on this user's OWN past data, what intervention should they take
right now and why?"""


def generate_coaching_recommendation(current_state: Dict[str, Any], retrieved_sessions: List[Dict[str, Any]]) -> str:
    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
    gemini_api_key = os.getenv("GEMINI_API_KEY")

    # Build a deterministic fallback from retrieved history so the dashboard remains useful
    # even when external API calls fail (billing, quota, network, etc.).
    intervention_stats: Dict[str, List[float]] = {}
    for row in retrieved_sessions:
        key = str(row.get("intervention_suggested", "breathing exercise"))
        intervention_stats.setdefault(key, []).append(float(row.get("recovery_pct", 0)))
    if intervention_stats:
        best_name = max(intervention_stats, key=lambda k: sum(intervention_stats[k]) / max(1, len(intervention_stats[k])))
        best_avg = sum(intervention_stats[best_name]) / max(1, len(intervention_stats[best_name]))
    else:
        best_name = "breathing exercise"
        best_avg = 0.0

    fallback_text = (
        "Live LLM coaching is temporarily unavailable. "
        f"Based on your retrieved session history, **{best_name}** is the strongest immediate option "
        f"(~{best_avg:.1f}% average recovery in similar sessions)."
    )

    user_prompt = _build_user_prompt(current_state, retrieved_sessions)

    # 1) Try Claude first if key + SDK are available.
    if anthropic_api_key and Anthropic is not None:
        try:
            client = Anthropic(api_key=anthropic_api_key)
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=220,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_prompt}],
            )
            text_parts = [b.text for b in response.content if getattr(b, "type", "") == "text"]
            if text_parts:
                return "\n".join(text_parts).strip()
        except Exception:
            pass

    # 2) Fallback to Gemini if key + SDK are available.
    if gemini_api_key and genai is not None:
        try:
            genai.configure(api_key=gemini_api_key)
            model = genai.GenerativeModel("gemini-1.5-flash")
            gemini_prompt = f"SYSTEM:\n{SYSTEM_PROMPT}\n\nUSER:\n{user_prompt}"
            response = model.generate_content(gemini_prompt)
            text = getattr(response, "text", "")
            if text:
                return text.strip()
        except Exception:
            pass

    return fallback_text
