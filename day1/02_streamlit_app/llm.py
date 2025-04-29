# llm.py
import time
import torch
from transformers import pipeline
import streamlit as st
from config import MODELS


@st.cache_resource
def get_pipeline(model_name: str):
    """任意の model_name でキャッシュされた pipeline を返す"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    st.info(f"Using device: {device}")
    try:
        # GPU で float16 モードでロードを試行
        return pipeline(
            "text-generation",
            model=model_name,
            model_kwargs={"torch_dtype": torch.float16},
            device=device
        )
    except RuntimeError as e:
        err = str(e).lower()
        if "out of memory" in err or "cuda" in err:
            st.warning("GPUメモリ不足のため、CPUモードでロードを試みます。")
            torch.cuda.empty_cache()
            try:
                return pipeline(
                    "text-generation",
                    model=model_name,
                    device="cpu"
                )
            except Exception as e2:
                st.error(f"CPUモードでもロードに失敗しました: {e2}")
                return None
        else:
            st.error(f"モデル '{model_name}' のロードに失敗しました: {e}")
            return None


def generate_response(pipe, user_question: str):
    """LLM を使って回答を生成し、(回答, 応答時間) を返す"""
    if pipe is None:
        return "モデルがロードされていないため、回答を生成できません。", 0.0

    start = time.time()
    try:
        outputs = pipe(
            [{"role": "user", "content": user_question}],
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )
        text = ""
        out = outputs[0].get("generated_text")
        if isinstance(out, str):
            idx = out.find(user_question) + len(user_question)
            text = out[idx:].strip()
        else:
            for msg in out:
                if msg.get("role") == "assistant":
                    text = msg.get("content", "").strip()
        if not text:
            text = "回答の抽出に失敗しました。"
    except Exception as e:
        st.error(f"回答生成中にエラーが発生しました: {e}")
        text = f"エラーが発生しました: {e}"
    elapsed = time.time() - start
    return text, elapsed
