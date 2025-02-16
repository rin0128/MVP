from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from ai_query import chain
from pydantic import BaseModel

app = FastAPI()

# CORSミドルウェアを追加（全てのオリジンを許可）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# バリデーションエラーのハンドリング
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        content={"error": "リクエストのバリデーションに失敗しました", "details": exc.errors()},
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY
    )

# POSTリクエスト用のリクエストボディ定義
class QueryRequest(BaseModel):
    question: str

@app.get("/")
async def root():
    return {"message": "Hello FastAPI!"}

# GETリクエストは削除し、POSTリクエストで対応
@app.post("/ask")
async def ask_question(request: QueryRequest):
    question = request.question  # リクエストボディから質問を取得
    print(f"受信した質問: {question}")  # デバッグ用ログ
    try:
        result = chain.invoke({"question": question})  
        print(f"生成された回答: {result}")
        return {"answer": result}
    except Exception as e:
        print(f"エラー: {str(e)}")
        return JSONResponse(content={"error": str(e)}, status_code=500)
