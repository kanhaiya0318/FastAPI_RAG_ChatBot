import uuid
import time
import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000"
MAX_CHARS = 200 

st.set_page_config(page_title="FastAPI Chatbot", page_icon="💬")
st.title("💬 FastAPI Chatbot")


# ---------- helpers ----------
def create_chat() -> str:
    sid = str(uuid.uuid4())
    st.session_state.chats[sid] = {"title": "New chat", "messages": []}
    return sid


def new_chat():
    # don't pile up empty chats
    current = st.session_state.chats[st.session_state.current]
    if current["messages"]:
        st.session_state.current = create_chat()


def switch_chat(sid: str):
    st.session_state.current = sid


def delete_chat(sid: str):
    """Delete one chat on the backend and in the UI."""
    try:
        requests.post(f"{API_URL}/sessions/{sid}/end", timeout=10)
    except requests.RequestException:
        pass  # 404 / backend down: still remove it from the UI
    st.session_state.chats.pop(sid, None)
    if st.session_state.current == sid:
        remaining = list(st.session_state.chats)
        st.session_state.current = remaining[-1] if remaining else create_chat()

def chat_stream(session_id: str, prompt: str, thinking_mode:str):
    try:
        with requests.post(
            f"{API_URL}/chat/message_stream",
            json={"session_id":session_id,"thinking_mode":thinking_mode, "message": prompt},
            stream=True,
            timeout = (20.0, 20.0)             # (connect, time bw 2 chunks)
        ) as resp:
            resp.raise_for_status()
            resp.encoding = "utf-8"
            for piece in resp.iter_content(chunk_size = None ,decode_unicode=True):
                if piece:
                    yield piece
    except requests.ConnectionError:
        yield "⚠️ Can't reach the backend. Is uvicorn running on port 8000?"
    except requests.HTTPError as e:
        yield f"⚠️ Backend error ({e.response.status_code})"
    except requests.RequestException as e:
        yield f"⚠️ Request failed: {e}"


def chat_n(session_id: str, prompt:str):
    try:
        start = time.perf_counter()
        resp = requests.post(
            f"{API_URL}/chat/send",
            json={"session_id": current_id, "message": prompt},
            timeout=60,
        )
        st.caption(f"⏱ {time.perf_counter() - start:.1f}s")
        resp.raise_for_status()
        answer = resp.json()["response"]
        return answer
    except requests.ConnectionError:
        return "⚠️ Can't reach the backend. Is uvicorn running on port 8000?"
    except requests.HTTPError:
        detail = resp.json().get("detail", resp.text)
        return f"⚠️ Backend error ({resp.status_code}): {detail}"
    except requests.RequestException as e:
        return f"⚠️ Request failed: {e}"


def timed(gen):
    t0 = time.perf_counter(); first = None
    for piece in gen:
        if first is None:
            first = time.perf_counter() - t0
        yield piece
    print(f"[ui] first word {first:.1f}s | total {time.perf_counter()-t0:.1f}s")



# ---------- session state ----------
if "chats" not in st.session_state:
    st.session_state.chats = {}
    st.session_state.current = create_chat()

current_id = st.session_state.current
chat = st.session_state.chats[current_id]

# ---------- sidebar ----------
with st.sidebar:
    st.button("➕ New chat", on_click=new_chat, use_container_width=True)
    st.divider()
    st.selectbox("🧠 Thinking", ["minimal", "low", "medium", "high"], key="thinking_mode")
    st.subheader("Chats")
    for sid, c in reversed(list(st.session_state.chats.items())):
        col1, col2 = st.columns([5, 1])
        col1.button(
            c["title"],
            key=f"open_{sid}",
            on_click=switch_chat,
            args=(sid,),
            type="primary" if sid == current_id else "secondary",
            use_container_width=True,
        )
        col2.button("🗑️", key=f"del_{sid}", on_click=delete_chat, args=(sid,))

# ---------- render history ----------
for msg in chat["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------- handle new input ----------
if prompt := st.chat_input("Type your message...", max_chars=MAX_CHARS):
    if not chat["messages"]:
        chat["title"] = prompt[:30] + ("…" if len(prompt) > 30 else "")
    chat["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # with st.spinner("Thinking..."):
        #     answer = chat_n(session_id, prompt)
        # st.markdown(answer)
        answer = st.write_stream(timed(chat_stream(current_id, prompt, st.session_state.thinking_mode)))
        

    chat["messages"].append({"role": "assistant", "content": answer})
    st.rerun()  # refresh sidebar so the new title shows