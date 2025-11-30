import os
import tempfile

import streamlit as st
from PyPDF2 import PdfReader
from docx import Document


def pdf_to_docx_simple(pdf_path: str, docx_path: str):
    reader = PdfReader(pdf_path)
    doc = Document()

    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            for line in text.splitlines():
                doc.add_paragraph(line)
        if i < len(reader.pages) - 1:
            doc.add_page_break()

    doc.save(docx_path)


def main():
    st.set_page_config(page_title="PDF → DOCX 변환기 (텍스트만)", page_icon="📄")
    st.title("📄 PDF를 DOCX로 변환하기 (텍스트만 추출)")
    st.write("레이아웃·이미지는 무시하고, PDF 안의 텍스트만 DOCX 파일로 변환합니다.")

    uploaded_file = st.file_uploader("PDF 파일을 업로드하세요", type=["pdf"])

    if st.button("변환 시작"):
        if uploaded_file is None:
            st.warning("먼저 PDF 파일을 업로드해주세요.")
            return

        with st.spinner("PDF를 처리하는 중입니다..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
                tmp_pdf.write(uploaded_file.read())
                pdf_path = tmp_pdf.name

            base_name = os.path.splitext(os.path.basename(uploaded_file.name))[0]
            with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp_docx:
                docx_path = tmp_docx.name

            try:
                pdf_to_docx_simple(pdf_path, docx_path)

                with open(docx_path, "rb") as f:
                    docx_data = f.read()

                st.success("변환이 완료되었습니다!")
                st.download_button(
                    label="DOCX 파일 다운로드",
                    data=docx_data,
                    file_name=f"{base_name}.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                )

            except Exception as e:
                st.error(f"변환 중 오류가 발생했습니다: {e}")

            finally:
                try:
                    os.remove(pdf_path)
                except Exception:
                    pass
                try:
                    os.remove(docx_path)
                except Exception:
                    pass


if __name__ == "__main__":
    main()
