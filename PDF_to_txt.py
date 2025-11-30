import io
import re
import zipfile
from datetime import datetime

import nltk
from spellchecker import SpellChecker
import streamlit as st

# -------------------- NLTK 준비 -------------------- #
def _ensure_nltk():
    """NLTK 데이터 다운로드 및 검증"""
    required_packages = ["punkt", "punkt_tab"]

    for package in required_packages:
        try:
            nltk.data.find(f"tokenizers/{package}")
        except (LookupError, OSError):
            try:
                nltk.download(package, quiet=True)
            except Exception as e:
                if package == "punkt":
                    raise Exception(f"Failed to download required NLTK data: {e}")
                else:
                    print(f"Warning: Could not download {package}, continuing anyway...")


_ensure_nltk()

# -------------------- 핵심 함수 -------------------- #
_WORD_RE = re.compile(r"^[A-Za-z][A-Za-z'-]*$")


def analyze_spelling(text, spell_checker):
    words = nltk.word_tokenize(text)
    tokens = [w for w in words if _WORD_RE.match(w)]
    lowers = [w.lower() for w in tokens]
    misspelled = spell_checker.unknown(lowers)
    corrections = {w: spell_checker.correction(w) for w in misspelled}
    return corrections, len(misspelled)


def correct_spelling(text, spell_checker):
    words = nltk.word_tokenize(text)
    out = []

    for tok in words:
        if _WORD_RE.match(tok):
            # 전부 대문자인 단어(약어 등)는 건드리지 않기 (선택적 정책)
            if tok.isupper():
                out.append(tok)
                continue

            corr = spell_checker.correction(tok.lower()) or tok
            if tok[:1].isupper():
                corr = corr.capitalize()
            out.append(corr)
        else:
            out.append(tok)

    s = " ".join(out)

    # 구두점 앞 공백 제거
    for p in [",", ".", "!", "?", ":", ";"]:
        s = s.replace(f" {p}", p)

    return s


def decode_bytes_with_fallback(data: bytes) -> str:
    """여러 인코딩 시도해서 텍스트 디코딩"""
    for enc in ("utf-8", "cp949", "euc-kr", "latin-1"):
        try:
            return data.decode(enc)
        except (UnicodeDecodeError, LookupError):
            continue
    # 마지막 수단: latin-1로 강제 디코딩
    return data.decode("latin-1", errors="replace")


# -------------------- Streamlit UI -------------------- #
st.set_page_config(
    page_title="YONSEI SPELLING DETECT TOOL",
    layout="wide",
)

st.title("YONSEI SPELLING DETECT TOOL")
st.caption("Batch English spelling detection & correction tool for .txt files (Yonsei ver.)")

st.markdown(
    """
1. 좌측에서 하나 이상의 `.txt` 파일을 업로드합니다.  
2. **Run Spell Check** 버튼을 누르면 스펠링 오류를 분석하고 교정합니다.  
3. 결과:
   - 중복 제거된 스펠링 오류 목록을 화면에 표시
   - CSV로 다운로드
   - 교정된 텍스트를 ZIP 파일로 묶어 다운로드
"""
)

uploaded_files = st.file_uploader(
    "📂 Upload .txt files",
    type=["txt"],
    accept_multiple_files=True,
    help="여러 개의 .txt 파일을 동시에 업로드할 수 있습니다.",
)

run_button = st.button("🚀 Run Spell Check")

if run_button:
    if not uploaded_files:
        st.error("적어도 하나의 .txt 파일을 업로드해 주세요.")
    else:
        spell = SpellChecker()

        dedup = {}
        corrected_files = {}  # filename -> corrected_text
        total_miss_count = 0

        progress_bar = st.progress(0)
        status_text = st.empty()

        num_files = len(uploaded_files)

        for idx, uploaded in enumerate(uploaded_files, start=1):
            filename = uploaded.name
            raw_bytes = uploaded.read()

            # 인코딩 처리
            text = decode_bytes_with_fallback(raw_bytes)

            # 스펠링 분석
            errors, miss_count = analyze_spelling(text, spell)
            total_miss_count += miss_count

            for w, c in errors.items():
                if w not in dedup:
                    dedup[w] = c

            # 교정
            fixed = correct_spelling(text, spell)
            corrected_files[filename] = fixed

            progress = int(idx / num_files * 100)
            progress_bar.progress(progress)
            status_text.text(f"Processing {idx}/{num_files} - {filename} (found {miss_count} errors)")

        # 결과 출력
        st.success(
            f"✅ 완료! {num_files}개 파일에서 총 {total_miss_count}개의 스펠링 오류를 발견했습니다.\n"
            f"중복 제거된 고유한 오류 수: {len(dedup)}개"
        )

        # dedup 표 보여주기
        if dedup:
            st.subheader("📋 Unique Spelling Errors (중복 제거)")
            table_data = [
                {"Spelling Error": err, "Correction": corr if corr else "(수정 불가)"}
                for err, corr in sorted(dedup.items())
            ]
            st.dataframe(table_data, use_container_width=True)

            # CSV 다운로드 버튼
            csv_buffer = io.StringIO()
            csv_buffer.write("스펠링 오류,올바른 단어\n")
            for err, corr in sorted(dedup.items()):
                fixed_corr = corr if corr else "(수정 불가)"
                # 콤마 처리(간단히 따옴표로 감싸기)
                csv_buffer.write(f"\"{err}\",\"{fixed_corr}\"\n")

            csv_bytes = csv_buffer.getvalue().encode("utf-8-sig")
            st.download_button(
                label="📊 Download Errors as CSV",
                data=csv_bytes,
                file_name=f"spelling_errors_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
            )

        # 교정된 텍스트 ZIP 다운로드
        if corrected_files:
            st.subheader("📦 Corrected Files Download")

            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                for fname, content in corrected_files.items():
                    # 교정된 텍스트를 UTF-8로 저장
                    zf.writestr(fname, content)

            zip_buffer.seek(0)

            st.download_button(
                label="📥 Download Corrected Files (ZIP)",
                data=zip_buffer,
                file_name=f"corrected_txt_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                mime="application/zip",
            )

        # 요약
        with st.expander("🔍 Summary"):
            st.write(f"- 처리한 파일 수: **{num_files}개**")
            st.write(f"- 총 발견된 스펠링 오류(중복 포함): **{total_miss_count}개**")
            st.write(f"- 고유한 스펠링 오류 수(중복 제거): **{len(dedup)}개**")
