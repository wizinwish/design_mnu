# 목포대 API Gateway와 통신하여 규칙 검색 결과를 포함한 야구 답변을 생성하는 모듈

import os
from typing import Any
from dotenv import load_dotenv
from openai import OpenAI, APIError, APITimeoutError
from app.search import search_engine

load_dotenv()

# 야구 초보자를 위한 기본 시스템 프롬프트 상수 (규칙 근거 지침 포함)
SYSTEM_PROMPT = (
    "너는 야구를 처음 보는 사람에게 규칙과 판정을 쉽게 설명하는 도우미야. "
    "어려운 용어는 풀어서 설명하고, 3~5문장으로 답해. "
    "제공된 참고 규칙을 근거로 답하고, 참고 규칙에 없는 내용은 일반 지식임을 밝혀라."
)

MOKPO_API_KEY = os.getenv("MOKPO_API_KEY")
MOKPO_BASE_URL = os.getenv("MOKPO_BASE_URL")
MOKPO_MODEL = os.getenv("MOKPO_MODEL")


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

    # 사용자의 마지막 질문 텍스트 추출
    user_queries = [m["content"] for m in messages if m.get("role") == "user"]
    last_query = user_queries[-1] if user_queries else ""

    # 질문과 관련된 상위 4개 규칙 청크 검색
    relevant_chunks = search_engine.search(last_query, top_k=4)

    # 검색된 규칙 내용을 시스템 프롬프트에 주입
    if relevant_chunks:
        rules_context = "\n\n[참고 규칙]"
        for i, chunk in enumerate(relevant_chunks, start=1):
            rules_context += f"\n{i}. 출처: {chunk['source']} (p.{chunk['page']})\n내용: {chunk['text']}"
        full_system_prompt = SYSTEM_PROMPT + rules_context
    else:
        full_system_prompt = SYSTEM_PROMPT

    sources = [
        {"source": chunk["source"], "page": chunk["page"]}
        for chunk in relevant_chunks
    ]

    try:
        client = OpenAI(
            api_key=MOKPO_API_KEY,
            base_url=MOKPO_BASE_URL,
            timeout=30.0,
        )

        combined_messages = [{"role": "system", "content": full_system_prompt}] + messages

        # 목포대 API Gateway로 채팅 완료 요청 전송
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
