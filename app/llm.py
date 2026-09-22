# 목포대 API Gateway와 통신하여 야구 규칙 답변을 생성하는 모듈

import os
from dotenv import load_dotenv
from openai import OpenAI, APIError, APITimeoutError

load_dotenv()

# 야구 초보자를 위한 기본 시스템 프롬프트 상수
SYSTEM_PROMPT = "너는 야구를 처음 보는 사람에게 규칙과 판정을 쉽게 설명하는 도우미야. 어려운 용어는 풀어서 설명하고, 3~5문장으로 답해."

MOKPO_API_KEY = os.getenv("MOKPO_API_KEY")
MOKPO_BASE_URL = os.getenv("MOKPO_BASE_URL")
MOKPO_MODEL = os.getenv("MOKPO_MODEL")


def generate_baseball_response(messages: list[dict[str, str]]) -> str:
    """사용자의 대화 기록을 받아 목포대 API Gateway에 전달하고 야구 규칙 답변을 반환합니다.

    Args:
        messages: 사용자 및 도우미의 대화 기록 목록 (예: [{"role": "user", "content": "스트라이크가 뭐야?"}])

    Returns:
        str: 게이트웨이에서 생성한 답변 문자열 또는 사용자 안내용 오류 메시지
    """
    if not MOKPO_API_KEY or not MOKPO_BASE_URL or not MOKPO_MODEL:
        return "API 설정(키, URL, 모델)이 누락되었습니다. .env 설정을 확인해주세요."

    try:
        client = OpenAI(
            api_key=MOKPO_API_KEY,
            base_url=MOKPO_BASE_URL,
            timeout=30.0,
        )

        # 시스템 프롬프트와 사용자의 이전 대화 기록을 순서대로 병합
        combined_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + messages

        # 목포대 API Gateway로 채팅 완료 요청 전송
        response = client.chat.completions.create(
            model=MOKPO_MODEL,
            messages=combined_messages,
        )

        return response.choices[0].message.content or "답변을 생성하지 못했습니다."

    except APITimeoutError:
        return "답변 요청 시간이 초과되었습니다(30초). 잠시 후 다시 시도해주세요."
    except APIError as e:
        return f"AI 서비스 응답 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요. ({e.message})"
    except Exception as e:
        return f"네트워크 또는 시스템 오류가 발생했습니다. 잠시 후 다시 질문해주세요. ({str(e)})"
