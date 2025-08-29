# streamlit_app.py
# Prompt Enhancer — Streamlit + OpenAI Responses API
# Updated: XML/JSON now output only the enhanced prompt. Also, enhanced prompt is rewritten to be more elaborate than raw instructions.

import json
import os
import html
from typing import Optional

import streamlit as st

try:
    from openai import OpenAI  # type: ignore
except Exception:
    OpenAI = None

APP_TITLE = "✨ Saket's Prompt Enhancer"
DEFAULT_MODEL = "gpt-4.1-mini"
ALT_MODELS = ["gpt-4.1", "gpt-4.1-mini"]

st.set_page_config(page_title="Prompt Enhancer", page_icon="✨", layout="centered")
st.title(APP_TITLE)
st.caption(
    "This app rewrites your inputs (Role, Context, Task) into a stronger, more elaborate prompt. "
    "It never executes the task — it only returns the improved prompt."
)

# --- API Key Handling ---
with st.sidebar:
    st.subheader("OpenAI API Settings")
    api_key_input = st.text_input(
        "OpenAI API Key",
        type="password",
        placeholder="sk-...",
        help="Paste your API key here. This app does not log or persist your key.",
    )

    api_key: str = (api_key_input or os.getenv("OPENAI_API_KEY", "")).strip()

    model_name = st.selectbox("Model", ALT_MODELS, index=ALT_MODELS.index(DEFAULT_MODEL))
    temperature = st.slider("Creativity (temperature)", 0.0, 1.2, 0.5, 0.1)

# --- Main Form ---
with st.form("prompt_form", clear_on_submit=False):
    role = st.text_input("Role", value="You are an experienced Python developer.")
    context = st.text_area(
        "Context",
        value="I am a complete beginner who does not know coding but am learning to use AI for coding.",
        height=100,
    )
    task = st.text_area(
        "Task",
        value="Help me build a Python app in Streamlit that enhances prompts based on Role, Context, and Task.",
        height=120,
    )
    generate = st.form_submit_button("Generate Enhanced Prompt ✨")

# --- Utility functions ---

def _fallback_prompt(role: str, context: str, task: str) -> str:
    return f"""
You are tasked with acting as {role}.

Background:
{context}

Objective:
{task}

Your enhanced prompt should expand on these inputs with clear structure, numbered steps, and explanatory depth. It must:
- Summarize the role and context in richer detail.
- Translate vague tasks into explicit, step-by-step goals.
- Require GPT to clarify missing assumptions and ask 2–4 targeted questions before giving a final response.
- Insist GPT provides a structured, beginner-friendly explanation in plain language.
""".strip()


def _to_xml(enhanced: str) -> str:
    return (
        "<prompt>\n"
        f"  <enhanced>{html.escape(enhanced)}</enhanced>\n"
        "</prompt>"
    )


def _to_json(enhanced: str) -> str:
    payload = {"enhanced_prompt": enhanced}
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _clarification_guard(enhanced: str) -> str:
    must_have_phrases = ["Before responding", "clarifying questions", "assumptions"]
    lowered = enhanced.lower()
    if not all(p.lower() in lowered for p in must_have_phrases):
        guard = (
            "\n\n# Important Instructions for GPT (auto-appended)\n"
            "1) **Before responding**, list key assumptions.\n"
            "2) Ask 2–4 **clarifying questions** and wait for my answers first.\n"
        )
        return enhanced.strip() + guard
    return enhanced


def generate_with_openai(api_key: str, model: str, role: str, context: str, task: str, temperature: float) -> Optional[str]:
    if not api_key or OpenAI is None:
        return None
    try:
        client = OpenAI(api_key=api_key)
        system = (
            "You are an expert prompt engineer. Rewrite the user's inputs into a single, "
            "clear, elaborate, and actionable prompt for GPT. IMPORTANT: Do not execute the task. "
            "Return only the enhanced prompt itself. The enhanced prompt MUST instruct GPT "
            "to list key assumptions and ask clarifying questions BEFORE responding."
        )
        user_msg = (
            "Inputs given:\n"
            f"Role: {role}\n"
            f"Context: {context}\n"
            f"Task: {task}\n\n"
            "Rewrite into an expanded, polished prompt that enriches details, emphasizes clarity, "
            "and guides GPT to respond with structure and depth. The enhanced prompt should be more elaborate than the original instructions."
        )

        resp = client.responses.create(
            model=model,
            temperature=temperature,
            input=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_msg},
            ],
        )

        enhanced = getattr(resp, "output_text", "").strip()
        return enhanced or None
    except Exception as e:
        st.warning(f"OpenAI call skipped due to an error: {e}")
        return None


if generate:
    with st.spinner("Creating your enhanced prompt…"):
        enhanced = generate_with_openai(api_key, model_name, role, context, task, temperature)
        if not enhanced:
            st.info("Using offline template (no/invalid API key or network error).")
            enhanced = _fallback_prompt(role, context, task)
        enhanced = _clarification_guard(enhanced)

        tabs = st.tabs(["Plain text", "XML", "JSON"])
        with tabs[0]:
            st.code(enhanced, language="markdown")
        with tabs[1]:
            st.code(_to_xml(enhanced), language="xml")
        with tabs[2]:
            st.code(_to_json(enhanced), language="json")

        st.success("Done! Copy your preferred format above.")
