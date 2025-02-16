import logging
import re
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.graphs import Neo4jGraph
from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, OPENAI_API_KEY

# ログ設定
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Neo4j に接続
graph = Neo4jGraph(
    url=NEO4J_URI,
    username=NEO4J_USER,
    password=NEO4J_PASSWORD
)
logger.info("Neo4jに接続しました。")

# ======================
# 1. Cypherクエリ生成プロンプト
# ======================
cypher_template = """Neo4jの以下のグラフスキーマに基づいて、ユーザーの質問に答えるCypherクエリを書いてください。
スキーマ: {schema}
質問: {question}

以下に注意してください:
- クエリ以外の文章は出力しないでください。
- 質問がグラフと無関係、もしくはクエリ生成が不可能な場合は「NO_QUERY」だけを返してください。

Cypherクエリ:"""

cypher_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "入力された質問をCypherクエリに変換してください。クエリ以外は生成しないでください。"),
        ("human", cypher_template),
    ]
)
logger.debug("Cypherクエリ生成プロンプトを設定しました。")

# ======================
# 2. OpenAI LLM のセットアップ
# ======================
llm = ChatOpenAI(
    model="gpt-4o",
    temperature=1,
    openai_api_key=OPENAI_API_KEY
)
logger.debug("OpenAI LLMを初期化しました。")

# ======================
# 3. クエリ生成チェイン
# ======================
queryGenChain = (
    RunnablePassthrough.assign(schema=lambda _: graph.get_schema)  # graph.get_schema を schema にセット
    | cypher_prompt  
    | llm  
    | StrOutputParser()
)
logger.debug("クエリ生成チェインを構築しました。")

# ======================
# 4. クエリ実行 & 自然言語変換のチェイン
# ======================
response_template = """質問、Cypherクエリ、およびクエリ実行結果に基づいて、自然言語で回答を書いてください:
質問: {question}
Cypherクエリ: {query}
クエリ実行結果: {response}"""

response_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "入力された質問、クエリ、クエリ実行結果をもとに、自然言語の答えに変換してください。"),
        ("human", response_template),
    ]
)
logger.debug("自然言語変換用プロンプトを設定しました。")

# ======================
# 5. 全体チェイン
# ======================
def execute_cypher_or_none(query: str, graph: Neo4jGraph):
    """
    - 受け取ったクエリが 'NO_QUERY' なら None を返す
    - 正規表現で ```cypher などを除去して実行
    - 実行結果が空なら空のリスト/Noneを返す
    """
    logger.debug(f"execute_cypher_or_none: 受け取ったクエリ: {query}")
    if query.strip().upper() == "NO_QUERY":
        logger.info("クエリがNO_QUERYのため、グラフと無関係な質問と判断しました。")
        return None
    
    # 不要なマークダウン表記を除去
    cleaned_query = re.sub(r"```(cypher)?|```", "", query).strip()
    logger.debug(f"execute_cypher_or_none: 除去後のクエリ: {cleaned_query}")
    
    if not cleaned_query:
        logger.warning("クエリが空文字列になっています。")
        return None
    
    try:
        results = graph.query(cleaned_query)
        logger.debug(f"execute_cypher_or_none: クエリ実行結果: {results}")
        if not results:
            logger.info("クエリ実行結果が空でした。")
            return None
        return results
    except Exception as e:
        logger.exception(f"Neo4jクエリ実行時のエラー: {e}")
        return None

chain = (
    RunnablePassthrough.assign(query=queryGenChain)  
    | RunnablePassthrough.assign(
        response=lambda x: execute_cypher_or_none(
            query=x["query"],
            graph=graph
        )
    )
    | response_prompt  
    | llm  
    | StrOutputParser()
)
logger.debug("全体チェインを構築しました。")

# ----------------------
# FastAPI アプリの設定
# ----------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    question: str

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.error(f"リクエストバリデーションエラー: {exc.errors()}")
    return JSONResponse(
        content={"error": "リクエストのバリデーションに失敗しました", "details": exc.errors()},
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY
    )

@app.get("/")
async def root():
    logger.debug("ルートエンドポイントにアクセスがありました。")
    return {"message": "Hello FastAPI!"}

@app.post("/ask")
async def ask_question(request: QueryRequest):
    user_question = request.question
    logger.info(f"受信した質問: {user_question}")

    try:
        result = chain.invoke({"question": user_question})
        logger.info(f"生成された回答: {result}")
        return {"answer": result}
    except Exception as e:
        logger.exception(f"エラー発生: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)

# ----------------------
# ローカルデバッグ用: 直接実行時のテストコード
# ----------------------
if __name__ == "__main__":
    logger.info("[DEBUG MODE] ローカルデバッグ開始")
    test_question = "私は中尾です。私について投資家に魅力的に見えるように説明してください"
    logger.info(f"[DEBUG MODE] テスト用の質問: {test_question}")
    
    try:
        result = chain.invoke({"question": test_question})
        logger.info(f"[DEBUG MODE] 生成された回答: {result}")
        print("===== デバッグ結果 =====")
        print(f"質問: {test_question}")
        print(f"回答: {result}")
    except Exception as e:
        logger.exception("[DEBUG MODE] エラー発生")
