import os
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from langchain_classic.storage import LocalFileStore
from langchain_classic.embeddings.cache import CacheBackedEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_naver import ChatClovaX

from app.core.embeddings.clova import ClovaEmbeddings

load_dotenv()
clova_api_key = os.getenv("CLOVASTUDIO_API_KEY")


async def split_n_return_docs(text: str):
    document = Document(page_content=text)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=100
    )
    docs = splitter.split_documents([document])

    return docs


async def embedding_n_return_retriever(text: str):
    cache_dir = "./.cache/embeddings"
    local_embedding_store = LocalFileStore(cache_dir=cache_dir)

    clova_embeddings = ClovaEmbeddings(
        api_key=clova_api_key,
        endpoint=(
            "https://clovastudio.stream.ntruss.com/v1/api-tools/embedding/v2/"
        ),
    )
    docs = await split_n_return_docs(text)

    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
        clova_embeddings, local_embedding_store
    )
    vectorstore = FAISS.from_documents(docs, cached_embeddings)

    return vectorstore.as_retriever()


async def generate_summary(text: str):
    docs = await split_n_return_docs(text)

    chat = ChatClovaX(model="HCX-005")

    initial_prompt = PromptTemplate.from_template(
        """
            다음 문서를 요약해줘.
            ----------
            {context}
            ----------
        """
    )
    initial_chain = initial_prompt | chat | StrOutputParser()
    summary = initial_chain.invoke({"context": docs[0].page_content})

    summary_prompt = PromptTemplate.from_template(
        """
            기존의 요약본과 새로운 문서를 제공할거야.
            주어진 새로운 문서로 기존의 요약본을 조정해서 최종 요약본을 작성해줘.

            추가하거나 수정할 내용이 없다면 기존의 요약본을 그대로 반환해줘.
            주어진 문서만을 참고해야해. 임의의 내용을 반영해서는 안돼.
            또한, 부가적인 설명 없이 그저 요약본 그 자체만 반환해줘.


            기존의 요약본:
            {previous_summary}

            새로운 문서:
            ----------
            {context}
            ----------
        """
    )
    summary_chain = summary_prompt | chat | StrOutputParser()

    for doc in docs[1:]:
        print("===summary===")
        print(summary)
        summary = summary_chain.invoke(
            {"previous_summary": summary, "context": doc.page_content}
        )

    print("===final summary===")
    print(summary)
    return summary


async def respond_to_question(question: str, text: str):
    retriever = await embedding_n_return_retriever(text)

    chat = ChatClovaX(model="HCX-005")

    qna_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                    너는 유능한 AI 비서야.
                    너는 오로지 사용자가 제공한 문서들을 바탕으로 질문에 답변을 제공해야해.
                    만약 문서에 답이 없다면 "해당 질문에 대한 내용은 존재하지 않습니다."라고 대답해줘.
                    절대 임의로 내용을 생성하거나 추가하지 마.

                    문서: 
                    {context}

                """,
            ),
            ("human", "{question}"),
        ]
    )

    qna_chain = (
        {
            "context": retriever | RunnableLambda(text),
            "question": RunnablePassthrough(),
            # history can be added here in the future
        }
        | qna_prompt
        | chat
        | StrOutputParser()
    )

    answer = qna_chain.invoke(question)

    return answer
