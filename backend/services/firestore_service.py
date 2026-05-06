import uuid
from datetime import datetime, timezone
from typing import Optional

import firebase_admin
from firebase_admin import credentials, firestore

from models.firestore_models import Conversation, Message, MessageCreate

_db: Optional[firestore.Client] = None


def init_firestore(service_account_path: str) -> None:
    if not firebase_admin._apps:
        cred = credentials.Certificate(service_account_path)
        firebase_admin.initialize_app(cred)
    global _db
    _db = firestore.client()


def _db_client() -> firestore.Client:
    if _db is None:
        raise RuntimeError("Firestore not initialized. Call init_firestore() at startup.")
    return _db


def create_conversation(user_id: str) -> str:
    db = _db_client()
    conv_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)
    db.collection("conversations").document(conv_id).set({
        "id": conv_id,
        "user_id": user_id,
        "started_at": now,
        "ended_at": None,
        "status": "active",
    })
    return conv_id


def append_message(conv_id: str, message: MessageCreate) -> str:
    db = _db_client()
    msg_id = str(uuid.uuid4())
    msg_data = {
        "id": msg_id,
        "conversation_id": conv_id,
        "speaker": message.speaker,
        "text": message.text,
        "timestamp": message.timestamp,
        "emotion": message.emotion.model_dump() if message.emotion else None,
    }
    db.collection("conversations").document(conv_id).collection("messages").document(msg_id).set(msg_data)
    return msg_id


def get_recent_messages(conv_id: str, n: int = 10) -> list[Message]:
    db = _db_client()
    docs = (
        db.collection("conversations")
        .document(conv_id)
        .collection("messages")
        .order_by("timestamp", direction=firestore.Query.DESCENDING)
        .limit(n)
        .stream()
    )
    messages = []
    for doc in docs:
        data = doc.to_dict()
        emotion_raw = data.get("emotion")
        messages.append(Message(
            id=data["id"],
            conversation_id=data["conversation_id"],
            speaker=data["speaker"],
            text=data["text"],
            timestamp=data["timestamp"],
            emotion=emotion_raw if emotion_raw else None,
        ))
    # Return in chronological order
    return list(reversed(messages))
