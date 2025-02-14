from fastapi import FastAPI, Query, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from ai_query import chain  # 修正: query_pipeline → chain
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from ai_query import chain  # 修正：query_pipeline → chain
from pydantic import BaseModel

app = FastAPI()

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    print("🚨 Request Validation Error!")
    print(exc.errors())  # ❗詳細なエラーメッセージをターミナルに出力
    return JSONResponse(
        content={"error": "リクエストのバリデーションに失敗しました", "details": exc.errors()},
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY
    )

# CORSミドルウェアを追加（全てのオリジンを許可）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 必要なら特定のオリジンだけ許可
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    question: str

@app.get("/")
async def root():
    return {"message": "Hello FastAPI!"}

@app.get("/ask")
async def ask_question(
    question: str = Query(..., description="質問内容を入力")  # 修正: `""` → `...` (必須)
):
    print(f"受信した質問: {question}")  # デバッグログ追加
    try:
        result = chain.invoke({"question": question})  # 修正: query_pipeline → chain
        print(f" 生成された回答: {result}")  # パイプラインの出力を確認
        return {"answer": result}
    except Exception as e:
        print(f" エラー: {str(e)}")  # エラー発生時のログ
        return {"error": str(e)}