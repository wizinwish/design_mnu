# data 폴더의 야구 규칙 PDF 파일들을 읽어 문자 n-gram TF-IDF 인덱스를 구축하는 스크립트

import os
import pickle
from pathlib import Path
from typing import Any
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer


def split_text_into_chunks(
    text: str, source: str, page: int, chunk_size: int = 650, overlap: int = 100
) -> list[dict[str, Any]]:
    """페이지 텍스트를 500~800자 크기(겹침 100자)의 청크로 분할하고 메타데이터를 추가합니다.

    Args:
        text: 분할할 원본 페이지 텍스트
        source: 출처 PDF 파일명
        page: PDF 페이지 번호 (1-indexed)
        chunk_size: 목표 청크 글자 수 (기본 650자)
        overlap: 청크 간 중첩 글자 수 (기본 100자)

    Returns:
        list[dict[str, Any]]: 텍스트와 출처(파일명, 페이지)를 담은 청크 목록
    """
    clean_text = " ".join(text.split())
    if not clean_text:
        return []

    # 단일 페이지 텍스트가 청크 크기 이하이면 단일 청크로 유지
    if len(clean_text) <= chunk_size:
        return [{"text": clean_text, "source": source, "page": page}]

    chunks = []
    step = chunk_size - overlap
    for i in range(0, len(clean_text), step):
        chunk_str = clean_text[i : i + chunk_size]
        if len(chunk_str) < 100 and chunks:
            # 너무 짧은 마지막 조각은 제외
            continue
        chunks.append({"text": chunk_str, "source": source, "page": page})

    return chunks


def build_index(data_dir: str = "data", output_dir: str = "data/index") -> dict[str, Any]:
    """data 폴더 내 모든 PDF를 스캔해 청크를 만들고 TF-IDF 인덱스를 파일로 저장합니다.

    Args:
        data_dir: PDF 문서들이 위치한 디렉토리 경로
        output_dir: 생성된 인덱스 파일이 저장될 디렉토리 경로

    Returns:
        dict[str, Any]: 파일별 페이지 수, 총 청크 수, 인덱스 저장 경로 정보
    """
    data_path = Path(data_dir)
    pdf_files = sorted(list(data_path.glob("*.pdf")))

    if not pdf_files:
        print(f"[알림] {data_dir} 디렉토리에 PDF 파일이 없습니다.")
        return {"page_counts": {}, "total_chunks": 0, "output_file": None}

    all_chunks: list[dict[str, Any]] = []
    page_counts: dict[str, int] = {}

    for pdf_file in pdf_files:
        reader = PdfReader(str(pdf_file))
        total_pages = len(reader.pages)
        page_counts[pdf_file.name] = total_pages
        print(f"[PDF 분석] {pdf_file.name}: 총 {total_pages}페이지")

        # 각 페이지에서 텍스트를 추출하고 청크 분할
        for page_idx, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            page_chunks = split_text_into_chunks(text, pdf_file.name, page_idx)
            all_chunks.extend(page_chunks)

    print(f"[청크 생성 완료] 총 {len(all_chunks)}개 청크 생성됨")

    chunk_texts = [chunk["text"] for chunk in all_chunks]

    # 형태소 분석기 없이도 한국어 조사 및 어미 변화에 강한 유사도 검색을 수행하기 위해 문자 단위 n-gram 사용
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 3))
    tfidf_matrix = vectorizer.fit_transform(chunk_texts)

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    index_file = out_path / "index_data.pkl"

    with open(index_file, "wb") as f:
        pickle.dump(
            {
                "chunks": all_chunks,
                "vectorizer": vectorizer,
                "tfidf_matrix": tfidf_matrix,
            },
            f,
        )

    print(f"[인덱스 저장 완료] {index_file} ({os.path.getsize(index_file) / 1024 / 1024:.2f} MB)")

    return {
        "page_counts": page_counts,
        "total_chunks": len(all_chunks),
        "output_file": str(index_file),
    }


if __name__ == "__main__":
    result = build_index()
    print("\n--- 인덱스 구축 요약 ---")
    for fname, pcount in result["page_counts"].items():
        print(f"- {fname}: {pcount}페이지")
    print(f"총 생성 청크 수: {result['total_chunks']}개")
