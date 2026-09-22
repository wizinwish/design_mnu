# 저장된 TF-IDF 인덱스를 로드하여 사용자 질문과 관련된 규칙 청크를 검색하는 모듈

import pickle
from pathlib import Path
from typing import Any
from sklearn.metrics.pairwise import cosine_similarity


class RuleSearchEngine:
    """PDF 규칙 인덱스를 메모리에 유지하고 질의와 유사한 규칙 청크를 검색하는 엔진."""

    def __init__(self, index_file: str = "data/index/index_data.pkl"):
        """인덱스 파일을 로드하며, 파일이 없을 경우 경고 메시지를 출력하고 기본 모드로 준비합니다.

        Args:
            index_file: 사전 구축된 인덱스 pkl 파일 경로
        """
        self.index_file = Path(index_file)
        self.chunks: list[dict[str, Any]] = []
        self.vectorizer = None
        self.tfidf_matrix = None
        self.is_ready = False
        self.load_index()

    def load_index(self) -> None:
        """인덱스 pkl 파일이 존재하면 메모리에 적재하고, 없으면 경고를 출력합니다."""
        if not self.index_file.exists():
            print(f"[경고] 규칙 인덱스 파일({self.index_file})이 없습니다. 챗봇이 기본 지식 모드로 동작합니다.")
            self.is_ready = False
            return

        try:
            with open(self.index_file, "rb") as f:
                data = pickle.load(f)
                self.chunks = data["chunks"]
                self.vectorizer = data["vectorizer"]
                self.tfidf_matrix = data["tfidf_matrix"]
                self.is_ready = True
                print(f"[검색 엔진] 규칙 인덱스 로드 완료 (총 {len(self.chunks)}개 청크)")
        except Exception as e:
            print(f"[경고] 인덱스 로드 중 오류가 발생했습니다: {e}. 기본 지식 모드로 동작합니다.")
            self.is_ready = False

    def search(self, query: str, top_k: int = 4) -> list[dict[str, Any]]:
        """질문 문자열과 가장 유사도가 높은 상위 청크들을 검색합니다.

        Args:
            query: 사용자의 마지막 질문 문자열
            top_k: 반환할 최대 청크 개수 (기본 4개)

        Returns:
            list[dict[str, Any]]: 유사도 점수가 포함된 상위 규칙 청크 목록
        """
        if not self.is_ready or not query.strip():
            return []

        # 사용자 질의를 기존 어휘 사전에 맞춰 TF-IDF 벡터로 변환
        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()

        # 유사도 상위 k개의 인덱스 추출
        top_indices = similarities.argsort()[::-1][:top_k]

        results = []
        for idx in top_indices:
            score = float(similarities[idx])
            # 최소한의 유사도가 있는 청크만 채택 (0점 제외)
            if score > 0.0:
                chunk = self.chunks[idx].copy()
                chunk["score"] = score
                results.append(chunk)

        return results


# 서버 시작 시 한 번만 인덱스를 로드하는 전역 검색 엔진 인스턴스
search_engine = RuleSearchEngine()
