import io
import re
from datetime import datetime

import nltk
import pandas as pd
import streamlit as st
from spellchecker import SpellChecker

# -------------------- NLTK 준비 -------------------- #
def _ensure_nltk():
    """NLTK 데이터 다운로드 및 검증 (토크나이저 + 품사 태거)"""
    required_resources = [
        ("tokenizers/punkt", "punkt", True),
        ("tokenizers/punkt_tab", "punkt_tab", False),
        ("taggers/averaged_perceptron_tagger", "averaged_perceptron_tagger", True),
        ("taggers/averaged_perceptron_tagger_eng", "averaged_perceptron_tagger_eng", False),
    ]

    for path, package, required in required_resources:
        try:
            nltk.data.find(path)
        except (LookupError, OSError):
            try:
                st.write(f"Downloading NLTK '{package}'...")
                nltk.download(package, quiet=True)
            except Exception as e:
                if required:
                    raise Exception(f"Failed to download required NLTK data '{package}': {e}")
                else:
                    st.write(f"Warning: Could not download optional NLTK data '{package}': {e}")


_WORD_RE = re.compile(r"^[A-Za-z][A-Za-z'-]*$")


def analyze_spelling(text, spell_checker):
    """
    텍스트에서 스펠링 오류를 탐지하고,
    각 오류에 대해 (교정어, 품사 태그, 전체 오류 개수)를 반환.
    - corrections: {단어(lower): 교정어}
    - pos_map: {단어(lower): 품사 태그 문자열}
    """
    words = nltk.word_tokenize(text)
    tokens = [w for w in words if _WORD_RE.match(w)]
    lowers = [w.lower() for w in tokens]

    misspelled = spell_checker.unknown(lowers)

    tagged = nltk.pos_tag(tokens)

    pos_counts = {}
    for tok, tag in tagged:
        key = tok.lower()
        if key not in pos_counts:
            pos_counts[key] = {}
        pos_counts[key][tag] = pos_counts[key].get(tag, 0) + 1

    pos_map = {}
    for w in misspelled:
        tag_dict = pos_counts.get(w, {})
        if tag_dict:
            best_tag = max(tag_dict.items(), key=lambda x: x[1])[0]
            pos_map[w] = best_tag
        else:
            pos_map[w] = ""

    corrections = {w: spell_checker.correction(w) for w in misspelled}
    return corrections, pos_map, len(misspelled)


@st.cache_resource
def get_spellchecker():
    _ensure_nltk()
    return SpellChecker()


def main():
    st.set_page_config(
        page_title="YONSEI SPELLING DETECT TOOL",
        layout="wide",
    )

    st.title("YONSEI SPELLING DETECT TOOL")
    st.write(
        "여러 개의 `.txt` 파일을 업로드하면, 스펠링 오류와 품사(Word Class), 교정어를 한 번에 확인할 수 있습니다."
    )

    uploaded_files = st.file_uploader(
        "분석할 .txt 파일을 업로드하세요 (여러 개 선택 가능)", type=["txt"], accept_multiple_files=True
    )

    run = st.button("🚀 Run Spelling Detection")

    if run:
        if not uploaded_files:
            st.warning("먼저 .txt 파일을 하나 이상 업로드하세요.")
            return

        spell = get_spellchecker()
        all_rows = []

        progress = st.progress(0.0)
        total = len(uploaded_files)

        for idx, uploaded in enumerate(uploaded_files, start=1):
            raw = uploaded.read()

            text = None
            for enc in ("utf-8", "cp949", "euc-kr", "latin-1"):
                try:
                    text = raw.decode(enc)
                    break
                except UnicodeDecodeError:
                    continue

            if text is None:
                st.warning(f"⚠️ {uploaded.name} - 인코딩 오류로 건너뜀")
                progress.progress(idx / total)
                continue

            corrections, pos_map, miss_count = analyze_spelling(text, spell)

            for err, corr in corrections.items():
                all_rows.append(
                    {
                        "file": uploaded.name,
                        "spelling_error": err,
                        "word_class": pos_map.get(err, ""),
                        "correction": corr if corr else "",
                    }
                )

            progress.progress(idx / total)

        if not all_rows:
            st.info("스펠링 오류가 발견되지 않았거나, 분석 가능한 단어가 없습니다.")
            return

        df = pd.DataFrame(all_rows)
        st.subheader("Detected Spelling Errors")
        st.dataframe(df, use_container_width=True)

        csv = df.to_csv(index=False, encoding="utf-8-sig")
        st.download_button(
            label="📊 Download CSV",
            data=csv,
            file_name=f"yonsei_spelling_errors_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()
