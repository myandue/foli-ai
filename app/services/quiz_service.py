import json

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableMap
from langchain_core.output_parsers import JsonOutputParser
from langchain_community.retrievers import WikipediaRetriever

from langchain_naver import ChatClovaX


def format_docs(docs: list[Document]):
    return "\n\n".join(doc.page_content for doc in docs)


def search_wikipedia(topic: str):
    retriever = WikipediaRetriever(lang="ko")
    return retriever.invoke(topic)


def generate_quiz_by_docs(docs: list[Document], amount: int, level: str):
    chat = ChatClovaX(model="HCX-007")

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
            너는 퀴즈 제작 도우미야.
            아래 정보를 이용해 JSON 형식으로 퀴즈를 만들어야 해.

            각 퀴즈에는 **4개의 선택지**가 있어야 하며, 오직 하나의 선택지만이 정답이어야해.

            JSON의 스키마는 다음과 같아. property의 타입과 구조를 반드시 지켜야 해.:
            {output_schema}

            오로지 주어진 문서의 내용만을 바탕으로 퀴즈를 만들어야해.
            문서: {content}
            문항 수: {num_questions}
            난이도: {difficulty_level}

            output 형태와 문항 수를 절대적으로 지켜야 해.

            출력은 반드시 **순수 JSON만** 포함해야 해. json 블록(```json)을 절대 씌우지 마.
            설명이나 추가 텍스트는 절대 포함하지 마.
            """,
            ),
        ]
    )

    output_schema = {
        "questions": [
            {
                "question": "string",
                "answers": [
                    {"answer": "string", "correct": "boolean"},
                    {"answer": "string", "correct": "boolean"},
                    {"answer": "string", "correct": "boolean"},
                    {"answer": "string", "correct": "boolean"},
                ],
            }
        ]
    }

    chain = (
        RunnableMap(
            {
                "content": lambda x: format_docs(x["docs"]),
                "num_questions": lambda x: str(x["amount"]),
                "difficulty_level": lambda x: x["level"],
                "output_schema": lambda _: json.dumps(
                    output_schema, ensure_ascii=False
                ),
            }
        )
        | prompt
        | chat
        | JsonOutputParser()
    )

    return chain.invoke({"docs": docs, "amount": amount, "level": level})


def generate_quiz_by_keyword(keyword: str, amount: int, level: str):
    docs = search_wikipedia(topic=keyword)
    return generate_quiz_by_docs(docs, amount, level)
