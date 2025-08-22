"""Streamlit app demonstrating OpenAI embeddings via LangChain.

The app provides a settings modal where users can supply an OpenAI API
key. The key is persisted in the browser's local storage so it only needs
to be entered once. When a key is available an example embedding for
"Hello world" is generated and displayed.
"""

import streamlit as st
from langchain_openai import OpenAIEmbeddings
from streamlit_js_eval import streamlit_js_eval


st.title("Hello, World!")


def load_api_key() -> str:
    """Retrieve the stored API key from local storage."""
    api_key = streamlit_js_eval(
        js_expressions="localStorage.getItem('openai_api_key')",
        key="get_api_key",
    )
    return api_key or ""


def save_api_key(key: str) -> None:
    """Persist the API key to local storage."""
    streamlit_js_eval(
        js_expressions=f"localStorage.setItem('openai_api_key', {json.dumps(key)})",
        key="set_api_key",
    )


if "openai_api_key" not in st.session_state:
    st.session_state.openai_api_key = load_api_key()


def open_settings() -> None:
    st.session_state.show_settings = True


def close_settings() -> None:
    st.session_state.show_settings = False


st.button("Settings", on_click=open_settings)

if st.session_state.get("show_settings"):
    with st.modal("Settings"):
        st.text_input(
            "OpenAI API Key",
            value=st.session_state.openai_api_key,
            key="api_key_input",
            type="password",
        )

        col_save, col_close = st.columns(2)
        with col_save:
            if st.button("Save"):
                st.session_state.openai_api_key = st.session_state.api_key_input
                save_api_key(st.session_state.api_key_input)
                close_settings()
        with col_close:
            st.button("Close", on_click=close_settings)


api_key = st.session_state.openai_api_key
                input_key = st.session_state.api_key_input
                if is_valid_openai_api_key(input_key):
                    st.session_state.openai_api_key = input_key
                    save_api_key(input_key)
                    close_settings()
                else:
                    st.error("Invalid API key format. Please enter a valid OpenAI API key (starts with 'sk-' and is the correct length).")
        with col_close:
            st.button("Close", on_click=close_settings)


api_key = st.session_state.openai_api_key
if is_valid_openai_api_key(api_key):
    embedder = OpenAIEmbeddings(
        model="text-embedding-3-small", openai_api_key=api_key
    )

@st.cache_data(show_spinner="Generating embedding...")
def get_hello_world_embedding(api_key: str):
    embedder = OpenAIEmbeddings(
        model="text-embedding-3-small", openai_api_key=api_key
    )
    return embedder.embed_query("Hello world")

if api_key:
    vector = get_hello_world_embedding(api_key)
    st.write(vector)
else:
    st.info("Set your OpenAI API key in settings to generate embeddings.")
