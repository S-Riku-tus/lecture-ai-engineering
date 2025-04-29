# ui.py
import streamlit as st
from streamlit.runtime.scriptrunner import RerunException, RerunData
import pandas as pd
import time
from config import MODELS
from database import save_to_db, get_chat_history, get_db_count, clear_db
from llm import generate_response, get_pipeline
from data import create_sample_evaluation_data
from metrics import get_metrics_descriptions
from config import MODEL_NAME, MODEL_NAME_1


# --- チャットページのUI ---
def display_chat_page():
    """ChatGPT 風チャット UI を表示"""

    model_label = st.session_state.get("selected_model_label", list(MODELS.keys())[0])
    model_name  = MODELS[model_label]

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "pending_response" not in st.session_state:
        st.session_state.pending_response = False

    # 既存メッセージを表示
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # ユーザー入力
    user_input = st.chat_input("質問を入力してください")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.session_state.pending_response = True
        # ここで強制 rerun
        raise RerunException(RerunData())

    # 「生成中...」と回答生成
    if st.session_state.pending_response:
        with st.chat_message("assistant"):
            st.markdown("生成中...")

        pipe = get_pipeline(model_name)
        if pipe is None:
            st.error(f"モデル `{model_label}` のロードに失敗しました。")
            st.session_state.pending_response = False
            raise RerunException(RerunData())

        answer, _ = generate_response(pipe, st.session_state.messages[-1]["content"])
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.session_state.pending_response = False
        # 生成結果反映のため再度 強制 rerun
        raise RerunException(RerunData())


def display_feedback_form():
    """フィードバック入力フォームを表示する"""
    with st.form("feedback_form"):
        st.subheader("フィードバック")
        feedback_options = ["正確", "部分的に正確", "不正確"]
        feedback = st.radio(
            "回答の評価",
            feedback_options,
            key="feedback_radio",
            label_visibility='collapsed',
            horizontal=True
        )
        correct_answer = st.text_area(
            "より正確な回答（任意）",
            key="correct_answer_input",
            height=100
        )
        feedback_comment = st.text_area(
            "コメント（任意）",
            key="feedback_comment_input",
            height=100
        )
        submitted = st.form_submit_button("フィードバックを送信")
        if submitted:
            is_correct = 1.0 if feedback == "正確" else (0.5 if feedback == "部分的に正確" else 0.0)
            combined_feedback = feedback
            if feedback_comment:
                combined_feedback += f": {feedback_comment}"

            save_to_db(
                st.session_state.current_question,
                st.session_state.current_answer,
                combined_feedback,
                correct_answer,
                is_correct,
                st.session_state.response_time
            )
            st.session_state.feedback_given = True
            st.success("フィードバックが保存されました！")
            # フォーム送信後の再描画
            st.experimental_rerun()


# --- 履歴閲覧ページのUI ---
def display_history_page():
    st.subheader("チャット履歴と評価指標")
    history_df = get_chat_history()
    if history_df.empty:
        st.info("まだチャット履歴がありません。")
        return

    tab1, tab2 = st.tabs(["履歴閲覧", "評価指標分析"])
    with tab1:
        display_history_list(history_df)
    with tab2:
        display_metrics_analysis(history_df)


def display_history_list(history_df):
    st.write("#### 履歴リスト")
    filter_options = {
        "すべて表示": None,
        "正確なもののみ": 1.0,
        "部分的に正確なもののみ": 0.5,
        "不正確なもののみ": 0.0
    }
    display_option = st.radio(
        "表示フィルタ",
        options=filter_options.keys(),
        horizontal=True,
        label_visibility="collapsed"
    )
    filter_value = filter_options[display_option]
    filtered_df = history_df if filter_value is None else history_df[
        history_df["is_correct"].notna() & (history_df["is_correct"] == filter_value)
    ]
    if filtered_df.empty:
        st.info("選択した条件に一致する履歴はありません。")
        return

    # ページネーション
    items_per_page = 5
    total_items = len(filtered_df)
    total_pages = (total_items + items_per_page - 1) // items_per_page
    current_page = st.number_input(
        'ページ', min_value=1, max_value=total_pages, value=1, step=1
    )
    start = (current_page - 1) * items_per_page
    end = start + items_per_page
    for _, row in filtered_df.iloc[start:end].iterrows():
        with st.expander(f"{row['timestamp']} - Q: {row['question'][:50]}..."):
            st.markdown(f"**Q:** {row['question']}")
            st.markdown(f"**A:** {row['answer']}")
            st.markdown(f"**Feedback:** {row['feedback']}")
            if row['correct_answer']:
                st.markdown(f"**Correct A:** {row['correct_answer']}")
            st.markdown("---")
            cols = st.columns(3)
            cols[0].metric("正確性スコア", f"{row['is_correct']:.1f}")
            cols[1].metric("応答時間(秒)", f"{row['response_time']:.2f}")
            cols[2].metric("単語数", f"{row['word_count']}")
            cols = st.columns(3)
            cols[0].metric(
                "BLEU",
                f"{row['bleu_score']:.4f}" if pd.notna(row['bleu_score']) else "-"
            )
            cols[1].metric(
                "類似度",
                f"{row['similarity_score']:.4f}" if pd.notna(row['similarity_score']) else "-"
            )
            cols[2].metric(
                "関連性",
                f"{row['relevance_score']:.4f}" if pd.notna(row['relevance_score']) else "-"
            )
    st.caption(f"{total_items} 件中 {start+1} - {min(end, total_items)} 件を表示")


def display_metrics_analysis(history_df):
    st.write("#### 評価指標の分析")
    analysis_df = history_df.dropna(subset=['is_correct'])
    if analysis_df.empty:
        st.warning("分析可能な評価データがありません。")
        return

    accuracy_labels = {1.0: '正確', 0.5: '部分的に正確', 0.0: '不正確'}
    analysis_df['正確性'] = analysis_df['is_correct'].map(accuracy_labels)

    st.write("##### 正確性の分布")
    counts = analysis_df['正確性'].value_counts()
    if not counts.empty:
        st.bar_chart(counts)
    else:
        st.info("正確性データがありません。")

    st.write("##### 応答時間とその他の指標の関係")
    metric_options = [c for c in ["bleu_score","similarity_score","relevance_score","word_count"]
                      if c in analysis_df.columns and analysis_df[c].notna().any()]
    if metric_options:
        opt = st.selectbox("比較指標を選択", metric_options, key="metric_select")
        chart_df = analysis_df[['response_time', opt, '正確性']].dropna()
        if not chart_df.empty:
            st.scatter_chart(chart_df, x='response_time', y=opt, color='正確性')
        else:
            st.info(f"{opt} と応答時間の有効データがありません。")
    else:
        st.info("比較可能な指標データがありません。")

    st.write("##### 評価指標の統計")
    stats_cols = [c for c in ['response_time','bleu_score','similarity_score','word_count','relevance_score']
                  if c in analysis_df.columns and analysis_df[c].notna().any()]
    if stats_cols:
        st.dataframe(analysis_df[stats_cols].describe())
    else:
        st.info("統計情報を計算できる指標がありません。")

    st.write("##### 効率性スコア (正確性 / (応答時間 + 0.1))")
    analysis_df['efficiency_score'] = analysis_df['is_correct'] / (analysis_df['response_time'].fillna(0)+0.1)
    if not analysis_df.empty:
        top10 = analysis_df.sort_values('efficiency_score', ascending=False).head(10)
        if 'id' in top10:
            st.bar_chart(top10.set_index('id')['efficiency_score'])
        else:
            st.bar_chart(top10['efficiency_score'])
    else:
        st.info("効率性スコアを計算できません。")


# --- サンプルデータ管理ページのUI ---
def display_data_page():
    st.subheader("サンプル評価データの管理")
    count = get_db_count()
    st.write(f"現在のデータベースには {count} 件のレコードがあります。")

    c1, c2 = st.columns(2)
    with c1:
        if st.button("サンプルデータを追加", key="create_samples"):
            create_sample_evaluation_data()
            st.rerun()
    with c2:
        if st.button("データベースをクリア", key="clear_db_button"):
            if clear_db():
                st.rerun()

    st.subheader("評価指標の説明")
    for metric, desc in get_metrics_descriptions().items():
        with st.expander(metric):
            st.write(desc)
