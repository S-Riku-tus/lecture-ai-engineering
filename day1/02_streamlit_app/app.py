# app.py
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # TensorFlow ログ抑制

import streamlit as st
import ui
import database, metrics, data
from config import MODELS

# ─── ページ設定は最初に ─────────────────────────────
st.set_page_config(page_title="Gemma Chatbot", layout="wide")

# ─── CSS 調整 ────────────────────────────────────────
st.markdown("""
<style>
  /* 全体コンテナ上部余白を詰める */
  .block-container { padding-top: 0.5rem !important; }
  /* h2/h3 上マージンをゼロに */
  .stMarkdown h2, .stMarkdown h3 { margin-top: 0 !important; }
  /* カスタムタイトル用クラス */
  .custom-title {
    margin-top: 0.25rem !important;
    margin-bottom: 0.25rem !important;
    font-size: 1.5rem;
    font-weight: bold;
  }
</style>
""", unsafe_allow_html=True)

# ─── 初期化処理を一度だけ実行 ───────────────────────
if 'app_initialized' not in st.session_state:
    metrics.initialize_nltk()
    database.init_db()
    data.ensure_initial_data()
    st.session_state.app_initialized = True

# ─── サイドバー：ページ切り替え ─────────────────────
st.sidebar.title("ナビゲーション")
if 'page' not in st.session_state:
    st.session_state.page = "チャット"
st.sidebar.radio(
    "ページ選択",
    options=["チャット", "履歴閲覧", "サンプルデータ管理"],
    key="page"
)

# ─── サイドバー：モデル選択 ─────────────────────────
st.sidebar.markdown("---")
st.sidebar.selectbox(
    "モデルを選択",
    options=list(MODELS.keys()),
    key="selected_model_label",
    help="選択したモデルで回答を生成します"
)


# ─── メインタイトル ─────────────────────────────────
st.markdown('<div class="custom-title">Gemma 2 Chatbot with Feedback</div>', unsafe_allow_html=True)

# ─── ページごとに表示を切り替え ─────────────────────
if st.session_state.page == "チャット":
    ui.display_chat_page()
elif st.session_state.page == "履歴閲覧":
    ui.display_history_page()
else:
    ui.display_data_page()
