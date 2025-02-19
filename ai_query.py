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

# Conversation memory を利用するためのインポート
from langchain.memory import ConversationBufferMemory

# ファイルの先頭あたりで定義する
global_system_message = """\
あなたは優秀なAIアシスタントです。
ユーザーに対して、できるだけ丁寧でわかりやすく、少しフレンドリーに回答をしてください。
不明な点や曖昧な情報がある場合は、その旨を伝えた上で推測や追加の情報を求めてください。
"""


def generate_final_answer(query: str) -> str:
    # 会話履歴を取得
    history = conversation_memory.load_memory_variables({}).get("chat_history", "")
    if history:
        modified_query = f"【会話履歴】\n{history}\n【新しい質問】\n{query}"
    else:
        modified_query = query
    logger.debug(f"generate_final_answer: modified_query (before system message): {modified_query}")

    # ここでグローバルシステムメッセージを追加する
    modified_query = f"{global_system_message}\n{modified_query}"
    logger.debug(f"generate_final_answer: modified_query (after system message): {modified_query}")

    additional_instructions = "論理的根拠を示してください。また、複数の視点から答えてください。"

    if requires_external_info(modified_query):
        logger.info("外部情報が必要な質問と判定。RAGパイプラインを使用します。")
        # RAGパスの場合、追加指示も付け加える
        modified_query = f"{modified_query}\n{additional_instructions}"
        final_answer = chain.invoke({"question": modified_query})
    else:
        logger.info("外部情報が不要な質問と判定。シンプルなGPT生成を使用します。")
        # 単体GPT生成の場合は generate_plain_answer 内で追加指示を使っています
        final_answer = generate_plain_answer(modified_query)
    
    conversation_memory.save_context({"input": modified_query}, {"output": final_answer})
    return final_answer



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

# ----------------------
# 会話履歴を保持するメモリ（MVP用：インメモリ）
# ----------------------
conversation_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=False)

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
    top_p=0.9,
    openai_api_key=OPENAI_API_KEY
)
logger.debug("OpenAI LLMを初期化しました。")

# ======================
# 3. クエリ生成チェイン（既存のRAGチェイン）
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
response_template = """以下の情報を踏まえて回答してください。

【入力情報】
- 質問: {question}
- Cypherクエリ: {query}
- クエリ実行結果: {response}

【回答の指示】
1. まず、上記の入力情報を簡潔に要約してください。
2. 次に、要約を基に複数の視点から論理的根拠を示しながら詳細な説明を行ってください。
3. 最後に、明確な結論として回答をまとめてください。

以上の手順に従って、自然で流暢な回答を生成してください。"""


response_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "入力された質問、クエリ、クエリ実行結果をもとに、自然言語の答えに変換してください。"),
        ("human", response_template),
    ]
)
logger.debug("自然言語変換用プロンプトを設定しました。")

# ======================
# 5. 全体チェイン（既存のRAG処理）
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

# ======================
# 6. ゲーティングモジュール（外部情報が必要かどうかの判定）
# ======================
def requires_external_info(query: str) -> bool:
    # 簡易的なキーワードチェックおよび文字数で判定
    keywords = ["中尾", "私", "投資家", "市場", "数値", "最新"]
    if any(word in query for word in keywords):
        logger.debug("requires_external_info: キーワードにより外部情報が必要と判定")
        return True
    if len(query) > 100:
        logger.debug("requires_external_info: 長文により外部情報が必要と判定")
        return True
    return False

# ======================
# 7. 単体GPT生成（外部情報が不要な場合のシンプルな回答）
# ======================
def generate_plain_answer(query: str) -> str:
    additional_instructions = "論理的根拠を示してください。また、複数の視点から答えてください。"
    prompt = (
        f"ユーザーの質問: {query}\n"
        f"{additional_instructions}\n"
        "流暢で詳細な回答を生成してください。"
    )
    logger.debug(f"generate_plain_answer: prompt: {prompt}")
    try:
        result = llm.invoke({"prompt": prompt})
        logger.debug(f"generate_plain_answer: result: {result}")
        return result
    except Exception as e:
        logger.exception("generate_plain_answer: エラー発生")
        return "回答生成中にエラーが発生しました。"



# ======================
# 8. 最終回答生成関数（ゲーティングによる分岐＋会話履歴の統合）
# ======================
def generate_final_answer(query: str) -> str:
    # 会話履歴を取得
    history = conversation_memory.load_memory_variables({}).get("chat_history", "")
    if history:
        modified_query = f"【会話履歴】\n{history}\n【新しい質問】\n{query}"
    else:
        modified_query = query

    logger.debug(f"generate_final_answer: modified_query: {modified_query}")

    if requires_external_info(modified_query):
        logger.info("外部情報が必要な質問と判定。RAGパイプラインを使用します。")
        final_answer = chain.invoke({"question": modified_query})
    else:
        logger.info("外部情報が不要な質問と判定。シンプルなGPT生成を使用します。")
        final_answer = generate_plain_answer(modified_query)
    
    # 会話履歴に保存（入力と出力のペア）
    conversation_memory.save_context({"input": modified_query}, {"output": final_answer})
    return final_answer

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
        answer = generate_final_answer(user_question)
        logger.info(f"生成された回答: {answer}")
        return {"answer": answer}
    except Exception as e:
        logger.exception(f"エラー発生: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)

# ----------------------
# ローカルデバッグ用: 直接実行時のテストコード
# ----------------------
if __name__ == "__main__":
    logger.info("[DEBUG MODE] ローカルデバッグ開始")
    test_question = "私は中尾です。私の本質的な強みはなんだと思いますか？クエリの情報から推論してください"
    logger.info(f"[DEBUG MODE] テスト用の質問: {test_question}")
    
    try:
        final_answer = generate_final_answer(test_question)
        logger.info(f"[DEBUG MODE] 生成された回答: {final_answer}")
        print("===== デバッグ結果 =====")
        print(f"質問: {test_question}")
        print(f"回答: {final_answer}")
    except Exception as e:
        logger.exception("[DEBUG MODE] エラー発生")
