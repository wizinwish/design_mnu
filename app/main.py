# 야구 규칙 설명 챗봇 웹 애플리케이션의 FastAPI 서버 메인 모듈

from typing import Literal
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.llm import generate_baseball_response

app = FastAPI(title="야구 규칙 도우미 챗봇")

# 정적 파일 디렉토리만 안전하게 마운트 (프로젝트 루트 노출 차단)
app.mount("/static", StaticFiles(directory="static"), name="static")


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(..., min_length=1)


class ChatResponse(BaseModel):
    reply: str


@app.get("/")
def get_index() -> FileResponse:
    """메인 웹 채팅 페이지(index.html)를 반환합니다.

    Returns:
        FileResponse: static 디렉토리 내 index.html 파일 응답
    """
    return FileResponse("static/index.html")


@app.post("/api/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest) -> ChatResponse:
    """사용자의 메시지 목록을 받아 야구 규칙 설명 답변을 반환합니다.

    Args:
        request: 사용자 및 어시스턴트의 이전 대화 목록이 담긴 요청 객체

    Returns:
        ChatResponse: LLM 또는 시스템이 생성한 최종 답변
    """
    try:
        # 클라이언트 메시지 목록을 딕셔너리 형태로 변환하여 LLM 모듈로 전달
        raw_messages = [msg.model_dump() for msg in request.messages]
        reply_text = generate_baseball_response(raw_messages)
        return ChatResponse(reply=reply_text)
    except Exception as e:
        return ChatResponse(
            reply=f"서버 처리 중 일시적인 오류가 발생했습니다. 다시 시도해주세요. ({str(e)})"
        )
