import json
import os
from datetime import datetime
from typing import TypedDict, Any

import httpx
from langchain_core.tools import tool
from langchain_core.utils.function_calling import convert_to_openai_tool
from langgraph.graph import StateGraph, END

from services.firestore_service import (
    get_recent_messages as _fs_get_recent_messages,
)


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    transcript: str
    emotion: str
    confidence: float
    speaker_id: str
    conversation_history: list
    candidate_response: str
    audio_url: str
    conv_id: str
    user_id: str
    # Extra fields for endpoint compatibility and personality context
    personality_type: str
    personality_description: str
    all_responses: dict  # {low, medium, high, contradictory} returned to caller


# ---------------------------------------------------------------------------
# LangChain tools
# ---------------------------------------------------------------------------

@tool
def get_recent_messages(conv_id: str, n: int = 10) -> list:
    """Retrieve the n most recent messages from the conversation to understand context before responding."""
    print(f"[tool] get_recent_messages conv={conv_id} n={n}")
    try:
        msgs = _fs_get_recent_messages(conv_id, n)
        return [{"speaker": m.speaker, "text": m.text} for m in msgs]
    except Exception:
        return []


@tool
def get_user_preferences(user_id: str) -> dict:
    """Get the user's preferences including personality type and preferred communication style."""
    print(f"[tool] get_user_preferences user={user_id}")
    return {
        "personality_type": None,
        "communication_style": "balanced and natural",
        "voice_preference": None,
        "preferred_response_length": "medium",
    }


@tool
def get_current_time() -> str:
    """Returns current date, day of week, and local time. Use when response depends on time-of-day or recency."""
    print(f"[tool] get_current_time")
    return datetime.now().strftime('%A, %B %d, %Y at %I:%M %p')


@tool
def get_user_location() -> dict:
    """Returns the AAC user's current location. Use when geographic context shapes the response."""
    print(f"[tool] get_user_location")
    return {
        "city": "Del Mar",
        "state": "California",
        "country": "USA",
        "timezone": "America/Los_Angeles",
    }


_WMO_DESCRIPTIONS = [
    (range(0, 1),    "clear"),
    (range(1, 2),    "mainly clear"),
    (range(2, 3),    "partly cloudy"),
    (range(3, 4),    "overcast"),
    (range(45, 49),  "fog"),
    (range(51, 56),  "drizzle"),
    (range(56, 58),  "freezing drizzle"),
    (range(61, 66),  "rain"),
    (range(66, 68),  "freezing rain"),
    (range(71, 78),  "snow"),
    (range(80, 83),  "rain showers"),
    (range(85, 87),  "snow showers"),
    (range(95, 96),  "thunderstorm"),
    (range(96, 100), "thunderstorm with hail"),
]


def _wmo_to_description(code: int) -> str:
    for span, label in _WMO_DESCRIPTIONS:
        if code in span:
            return label
    return "unknown"


@tool
def get_weather(city: str = "Del Mar") -> dict:
    """Returns current weather. Use when weather context could shape the response."""
    print(f"[tool] get_weather city={city}")
    try:
        url = (
            "https://api.open-meteo.com/v1/forecast"
            "?latitude=32.96&longitude=-117.27"
            "&current=temperature_2m,weather_code,wind_speed_10m"
            "&temperature_unit=fahrenheit&wind_speed_unit=mph"
        )
        with httpx.Client(timeout=10.0) as client:
            resp = client.get(url)
            resp.raise_for_status()
            data = resp.json()["current"]
        return {
            "temperature_f": data["temperature_2m"],
            "description": _wmo_to_description(int(data["weather_code"])),
            "wind_mph": data["wind_speed_10m"],
        }
    except Exception:
        return {"error": "weather unavailable"}


_TOOLS = [get_recent_messages, get_user_preferences, get_current_time, get_user_location, get_weather]
_TOOLS_OAI = [convert_to_openai_tool(t) for t in _TOOLS]
_TOOL_MAP = {t.name: t for t in _TOOLS}


# ---------------------------------------------------------------------------
# Lava-proxied GPT call
# ---------------------------------------------------------------------------

def _call_gpt(messages: list[dict], use_tools: bool = True) -> dict:
    """POST to GPT-4o-mini through the Lava proxy. Returns the full OpenAI response dict."""
    url = f"{os.environ['LAVA_BASE_URL']}/forward?u=https://api.openai.com/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.environ['LAVA_FORWARD_TOKEN']}",
    }
    payload: dict[str, Any] = {
        "model": "gpt-4o-mini",
        "messages": messages,
        "temperature": 0.35,
        "max_tokens": 1000,
    }
    if use_tools:
        payload["tools"] = _TOOLS_OAI
        payload["tool_choice"] = "auto"

    with httpx.Client(timeout=30.0) as client:
        resp = client.post(url, headers=headers, json=payload)
        resp.raise_for_status()
        return resp.json()


# ---------------------------------------------------------------------------
# Node helpers
# ---------------------------------------------------------------------------

_FALLBACK_RESPONSES = {
    "low": "I understand.",
    "medium": "That makes sense.",
    "high": "That sounds great!",
    "contradictory": "Oh wonderful.",
}


def _parse_four_variants(content: str) -> dict:
    try:
        clean = content.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        parsed = json.loads(clean)
        return {
            "low": parsed.get("low", _FALLBACK_RESPONSES["low"]),
            "medium": parsed.get("medium", _FALLBACK_RESPONSES["medium"]),
            "high": parsed.get("high", _FALLBACK_RESPONSES["high"]),
            "contradictory": parsed.get("contradictory", _FALLBACK_RESPONSES["contradictory"]),
        }
    except (json.JSONDecodeError, AttributeError):
        return _FALLBACK_RESPONSES.copy()


# ---------------------------------------------------------------------------
# Graph nodes
# ---------------------------------------------------------------------------

def classify_emotion_node(state: AgentState) -> AgentState:
    print(f"[node] classify_emotion")
    # In the HTTP flow, emotion is set by the EEG WebSocket pipeline on the frontend
    # and passed in the request. In a future integrated turn, predict_emotion() runs here.
    return state


def retrieve_context_node(state: AgentState) -> AgentState:
    print(f"[node] retrieve_context")
    if not state.get("conv_id"):
        return {**state, "conversation_history": []}
    try:
        msgs = _fs_get_recent_messages(state["conv_id"], 10)
        history = [{"speaker": m.speaker, "text": m.text} for m in msgs]
    except Exception:
        history = []
    return {**state, "conversation_history": history}


def _format_history_line(speaker: str, text: str) -> str:
    label = "self" if speaker == "USER" else speaker
    return f"[{label}]: {text}"


def generate_response_node(state: AgentState) -> AgentState:
    print(f"[node] generate_response")
    history_text = "\n".join(
        _format_history_line(m["speaker"], m["text"])
        for m in state.get("conversation_history", [])
    )
    personality_context = ""
    if state.get("personality_type"):
        personality_context = (
            f"\nUser personality type: {state['personality_type']}"
            f" — {state.get('personality_description', '')}"
        )

    keyword_count = len([k for k in state.get("transcript", "").split() if k.strip()])
    if keyword_count <= 3:
        length_guidance = {"low": "3-7 words", "medium": "1 sentence (8-15 words)",
                           "high": "2 sentences (15-25 words)", "contradictory": "1 sentence (8-15 words)"}
    elif keyword_count <= 6:
        length_guidance = {"low": "5-10 words", "medium": "1-2 sentences (15-25 words)",
                           "high": "2-3 sentences (25-40 words)", "contradictory": "1-2 sentences (15-25 words)"}
    else:
        length_guidance = {"low": "8-12 words", "medium": "2 sentences (20-35 words)",
                           "high": "3-4 sentences (40-60 words)", "contradictory": "2 sentences (20-35 words)"}

    emotion = state.get("emotion", "neutral")
    system_content = f"""You are a sentence builder helping AAC (augmentative and alternative communication) users express themselves naturally.

Speaker labels (read carefully — this is who said what in the conversation history):
- [self] = the AAC device user. This is the person you are helping. They cannot easily speak on their own; they compose responses through this assistive system and we synthesize them via TTS. Your job is to generate candidate sentences for [self] to say.
- [speaker_1], [speaker_2], etc. = real human conversation partners whose voices were captured by the microphone and diarized. These are the people [self] is talking *to*.
- [speaker_unknown] = a partner whose voice could not be reliably fingerprinted; treat as a generic partner.
You always generate responses spoken BY [self] addressed TO the speakers. Never generate text for the speakers themselves.

IMPORTANT — use the available tools before generating:
1. Call get_recent_messages to fetch the latest conversation turns and understand full context
2. Call get_user_preferences to adapt vocabulary and style to this user{personality_context}

You also have awareness tools — call them ONLY when the conversation actually calls for that context, not on every turn:
- get_current_time: when the response depends on time-of-day, day-of-week, or recency (e.g. "is it late?", "what time is it?", "how long has it been?").
- get_user_location: when geographic context shapes the response (e.g. talking about local places, distances, "around here").
- get_weather: when weather could shape the response (e.g. "should I bring a jacket?", "is it raining?", planning outdoor activity).
Skip these tools entirely when the topic is unrelated; redundant calls add latency.

Then generate exactly 4 response variants as valid JSON with keys "low", "medium", "high", "contradictory".
Output ONLY valid JSON — no preamble, no markdown fences.

Length guidance: low={length_guidance['low']}, medium={length_guidance['medium']}, high={length_guidance['high']}, contradictory={length_guidance['contradictory']}
Energy styles: low=brief/understated, medium=natural, high=expressive/detailed, contradictory=sarcastic/ironic
Emotion: {emotion}"""

    user_content = f"""Conv ID: {state.get('conv_id', '')}
User ID: {state.get('user_id', '')}
Keywords: {state.get('transcript', '')}
Emotion: {emotion}
Recent context already retrieved: {history_text or '(none yet)'}

Generate the 4 variants. Call tools first if you need more context."""

    messages: list[dict] = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]

    # Tool-calling loop — max 3 rounds
    final_content = ""
    for _ in range(3):
        response = _call_gpt(messages, use_tools=True)
        choice = response["choices"][0]["message"]

        if choice.get("tool_calls"):
            messages.append({"role": "assistant", **{k: v for k, v in choice.items() if k != "role"}})
            for tc in choice["tool_calls"]:
                fn_name = tc["function"]["name"]
                fn_args = json.loads(tc["function"]["arguments"])
                tool_fn = _TOOL_MAP.get(fn_name)
                result = tool_fn.invoke(fn_args) if tool_fn else []
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": json.dumps(result),
                })
        else:
            final_content = choice.get("content", "")
            break

    all_responses = _parse_four_variants(final_content)
    return {**state, "all_responses": all_responses, "candidate_response": all_responses["medium"]}


def synthesize_speech_node(state: AgentState) -> AgentState:
    print(f"[node] synthesize_speech")
    # TTS is triggered by the user's energy-level selection in a separate /api/text-to-speech request.
    # This node is a placeholder for future inline synthesis.
    return {**state, "audio_url": ""}


# ---------------------------------------------------------------------------
# Build and compile graph
# ---------------------------------------------------------------------------

def _build_graph():
    builder = StateGraph(AgentState)
    builder.add_node("classify_emotion", classify_emotion_node)
    builder.add_node("retrieve_context", retrieve_context_node)
    builder.add_node("generate_response", generate_response_node)
    builder.add_node("synthesize_speech", synthesize_speech_node)

    builder.set_entry_point("classify_emotion")
    builder.add_edge("classify_emotion", "retrieve_context")
    builder.add_edge("retrieve_context", "generate_response")
    builder.add_edge("generate_response", "synthesize_speech")
    builder.add_edge("synthesize_speech", END)

    return builder.compile()


conversation_graph = _build_graph()
