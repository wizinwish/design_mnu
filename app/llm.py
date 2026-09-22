# 목포대 API Gateway와 통신하여 규칙 검색 결과를 포함한 야구 답변을 생성하는 모듈

import os
from typing import Any
from dotenv import load_dotenv
from openai import OpenAI, APIError, APITimeoutError
from app.search import search_engine

load_dotenv()

# 야구 초보자를 위한 기본 시스템 프롬프트 상수 (규칙 근거 및 자연스러운 답변 표현 지침 포함)
SYSTEM_PROMPT = (
    "너는 야구를 처음 보는 사람에게 규칙과 판정을 쉽게 설명하는 도우미야. "
    "어려운 용어는 풀어서 설명하고, 3~5문장으로 답해. "
    "제공된 참고 규칙을 근거로 답하고, 참고 규칙에 없는 내용은 일반 지식임을 밝혀라. "
    "답변에서 '[참고 규칙 N]'처럼 참고 규칙 번호를 언급하지 말고, 규칙 내용을 자연스럽게 설명해. "
    "참고 규칙에 없는 내용을 덧붙일 때만 '일반적으로는' 같은 표현으로 구분해."
)

MOKPO_API_KEY = os.getenv("MOKPO_API_KEY")
MOKPO_BASE_URL = os.getenv("MOKPO_BASE_URL")
MOKPO_MODEL = os.getenv("MOKPO_MODEL")


def expand_query_with_llm(client: OpenAI, query: str) -> str:
    """사용자 질문을 공식 야구 규칙 및 리그 규정 검색 키워드로 확장합니다.

    Args:
        client: 초기화된 OpenAI 클라이언트 객체
        query: 사용자의 마지막 질문 문자열

    Returns:
        str: 생성된 공식 용어 검색 키워드 문자열 (실패 시 빈 문자열)
    """
    if not query.strip():
        return ""

    prompt = (
        "이 질문을 야구 규칙집·리그 규정에서 쓰는 공식 용어로 바꾼 검색 키워드 5~8개를 공백으로 구분해 출력해. "
        "설명 없이 키워드만. "
        "규정집 표기대로 띄어쓰기를 유지해. "
        "규칙·규정집의 실제 조항 제목과 본문에서 사용하는 표현(예: '기회', '구단당 기회', '신청 기회' 등)을 우선적으로 포함해. "
        "단, 단순 인사말이나 잡담 등 야구 규칙과 무관한 질문이면 아무것도 출력하지 마."
    )

    try:
        # LLM을 통해 일상 표현을 공식 규정집 검색 키워드로 변환
        response = client.chat.completions.create(
            model=MOKPO_MODEL,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": query},
            ],
            max_tokens=60,
        )
        keywords = response.choices[0].message.content or ""
        keywords = " ".join(keywords.split())
        print(f"[질문 변환] '{query}' -> '{keywords}'")
        return keywords
    except Exception as e:
        print(f"[질문 변환 실패] 원래 질문으로 검색 진행 ({e})")
        return ""


def generate_baseball_response(
    messages: list[dict[str, str]]
) -> tuple[str, list[dict[str, Any]]]:
    """사용자 대화 기록 및 관련 규칙을 바탕으로 게이트웨이 답변과 출처 목록을 반환합니다.

    Args:
        messages: 사용자 및 도우미의 대화 기록 목록 (예: [{"role": "user", "content": "..."}])

    Returns:
        tuple[str, list[dict[str, Any]]]: (생성된 답변 텍스트 또는 오류 메시지, 참고한 규칙 출처 목록)
    """
    if not MOKPO_API_KEY or not MOKPO_BASE_URL or not MOKPO_MODEL:
        return ("API 설정(키, URL, 모델)이 누락되었습니다. .env 설정을 확인해주세요.", [])

    try:
        client = OpenAI(
            api_key=MOKPO_API_KEY,
            base_url=MOKPO_BASE_URL,
            timeout=30.0,
        )
    except Exception as e:
        return (f"API 클라이언트 초기화 중 오류가 발생했습니다: {e}", [])

    # 사용자의 마지막 질문 텍스트 추출
    user_queries = [m["content"] for m in messages if m.get("role") == "user"]
    last_query = user_queries[-1] if user_queries else ""

    # 검색 전 질문 변환 단계: 공식 용어 키워드 생성 (실패 시 원본 질문으로 폴백)
    keywords = expand_query_with_llm(client, last_query)
    search_query = f"{last_query} {keywords}".strip() if keywords else last_query

    # 확장된 쿼리로 상위 8개 규칙 청크 검색 (임계값 미만 시 빈 리스트 반환)
    relevant_chunks = search_engine.search(search_query, top_k=8)

    # 검색된 규칙 내용을 시스템 프롬프트에 주입
    if relevant_chunks:
        rules_context = "\n\n[참고 규칙]"
        for i, chunk in enumerate(relevant_chunks, start=1):
            rules_context += f"\n{i}. 출처: {chunk['source']} (p.{chunk['page']})\n내용: {chunk['text']}"
        full_system_prompt = SYSTEM_PROMPT + rules_context
        sources = [
            {"source": chunk["source"], "page": chunk["page"]}
            for chunk in relevant_chunks
        ]
    else:
        full_system_prompt = SYSTEM_PROMPT
        sources = []

    try:
        combined_messages = [{"role": "system", "content": full_system_prompt}] + messages

        # 목포대 API Gateway로 최종 답변 생성 요청 전송
        response = client.chat.completions.create(
            model=MOKPO_MODEL,
            messages=combined_messages,
        )

        reply_content = response.choices[0].message.content or "답변을 생성하지 못했습니다."
        return (reply_content, sources)

    except APITimeoutError:
        return ("답변 요청 시간이 초과되었습니다(30초). 잠시 후 다시 시도해주세요.", sources)
    except APIError as e:
        return (f"AI 서비스 응답 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요. ({e.message})", sources)
    except Exception as e:
        return (f"네트워크 또는 시스템 오류가 발생했습니다. 잠시 후 다시 질문해주세요. ({str(e)})", sources)
