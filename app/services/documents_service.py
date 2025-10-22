from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_naver import ChatClovaX


async def split_n_return_docs(text_file):
    content = await text_file.read()
    text = content.decode("utf-8")

    document = Document(page_content=text)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=100
    )
    docs = splitter.split_documents([document])

    return docs


async def generate_summary(text_file):
    docs = await split_n_return_docs(text_file)

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
