import csv
import os
from pathlib import Path

from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.history_aware_retriever import create_history_aware_retriever
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, FewShotChatMessagePromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient

from config import answer_examples

store = {}
_collection_checked = False


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


def _index_source_path() -> Path:
    return Path(__file__).resolve().parent / "oliveyoung-markdown-index"


def _load_index_rows() -> list[Document]:
    source_path = _index_source_path()
    if not source_path.exists():
        return []

    columns = [
        "row_id",
        "rank",
        "goods_no",
        "disp_cat_no",
        "brand",
        "title",
        "url",
        "image_url",
        "price",
        "sale_price",
        "benefit",
        "captured_at",
    ]
    documents = []

    with source_path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if not row:
                continue
            values = row + [""] * (len(columns) - len(row))
            item = dict(zip(columns, values))
            content = "\n".join(
                [
                    f"brand: {item['brand']}",
                    f"title: {item['title']}",
                    f"price: {item['price']}",
                    f"sale_price: {item['sale_price']}",
                    f"benefit: {item['benefit']}",
                    f"url: {item['url']}",
                    f"image_url: {item['image_url']}",
                    f"captured_at: {item['captured_at']}",
                ]
            )
            metadata = {
                "row_id": item["row_id"],
                "rank": item["rank"],
                "goods_no": item["goods_no"],
                "disp_cat_no": item["disp_cat_no"],
                "brand": item["brand"],
                "url": item["url"],
            }
            documents.append(Document(page_content=content, metadata=metadata))

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    return splitter.split_documents(documents)


def _ensure_collection(embedding, collection_name: str, qdrant_url: str) -> None:
    global _collection_checked
    if _collection_checked:
        return
    _collection_checked = True

    client = QdrantClient(url=qdrant_url)
    try:
        client.get_collection(collection_name)
        return
    except Exception:
        pass

    documents = _load_index_rows()
    if not documents:
        return

    QdrantVectorStore.from_documents(
        documents=documents,
        embedding=embedding,
        url=qdrant_url,
        collection_name=collection_name,
    )


def get_retriever():
    embedding = OpenAIEmbeddings(model='text-embedding-3-large')
    collection_name = 'oliveyoung-markdown-index'
    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    _ensure_collection(embedding, collection_name, qdrant_url)
    database = QdrantVectorStore.from_existing_collection(
        collection_name=collection_name,
        embedding=embedding,
        url=qdrant_url,
    )
    retriever = database.as_retriever(search_kwargs={'k': 4})
    return retriever


def get_history_retriever():
    llm = get_llm()
    retriever = get_retriever()

    contextualize_q_system_prompt = (
        "주어진 대화 기록과 최신 질문을 참고하여, "
        "제공된 '올리브영 랭킹 데이터' 내에서 검색 가능한 구체적인 질문으로 재구성해주세요. "
        "질문의 '이거', '저거', '상위권 제품들' 같은 대명사는 "
        "'랭킹 1~10위 상품', '특정 브랜드(예: 메디큐브) 상품' 등으로 구체화해야 합니다. "
        "질문에 답변하지 말고 재구성된 질문만 반환하세요."
    )

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )
    return history_aware_retriever


def get_llm(model='gpt-4o-mini'):
    llm = ChatOpenAI(model=model)
    return llm


# def get_dictionary_chain():
#     dictionary = ["사람을 나타내는 표현 -> 거주자"]
#     llm = get_llm()
#     prompt = ChatPromptTemplate.from_template(f"""
#         사용자의 질문을 보고, 우리의 사전을 참고해서 사용자의 질문을 변경해주세요.
#         만약 변경할 필요가 없다고 판단된다면, 사용자의 질문을 변경하지 않아도 됩니다.
#         그런 경우에는 질문만 리턴해주세요
#         사전: {dictionary}
#
#         질문: {{question}}
#     """)
#
#     dictionary_chain = prompt | llm | StrOutputParser()
#
#     return dictionary_chain


def get_rag_chain():
    llm = get_llm()
    example_prompt = ChatPromptTemplate.from_messages(
        [
            ("human", "{input}"),
            ("ai", "{answer}"),
        ]
    )
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=answer_examples,
    )
    system_prompt = (
        "당신은 올리브영 랭킹 데이터를 분석하는 '마켓 인사이트 전문가'입니다. "
        "아래 제공된 CSV 포맷의 [실시간 랭킹 데이터]를 기반으로 사용자의 질문에 답변해주세요.\n"
        "데이터의 각 열(Column)은 순서대로 [순위, 식별자, 상품코드, 카테고리, 브랜드, 상품명, URL, 이미지, 정가, 할인가, 태그, 수집일시]를 의미합니다.\n\n"
        "답변 시 다음 원칙을 반드시 준수하세요:\n"
        "1. **트렌드 분석**: '특징', '공통점'을 물을 경우, 상품명에 포함된 키워드(예: PDRN, 어성초, 1+1, 기획, 리필)를 분석하여 답변하세요.\n"
        "2. **가격 정보**: 가격 관련 질문 시 '정가'와 '할인가'를 비교하여 할인율이나 가격 메리트를 언급하세요.\n"
        "3. **정확한 출처**: 답변은 반드시 제공된 데이터 리스트에 존재하는 상품에 한해서만 제공하고, 데이터에 없다면 모른다고 답하세요.\n"
        "4. **답변 스타일**: 마케터가 보고서에 쓰기 좋은 핵심 요약 형태로 2~3문장으로 답변하세요."
        "\n\n"
        "Thinking Process:\n"
        "- 사용자가 '메디큐브'를 물으면 브랜드 컬럼에서 '메디큐브'를 필터링한다.\n"
        "- '트렌드'를 물으면 상위 랭킹 상품들의 제목에서 반복되는 단어(예: 흔적, 잡티, 더블기획)를 찾는다.\n"
        "\n\n"
        "{context}"
    )

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            few_shot_prompt,
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = get_history_retriever()
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    ).pick('answer')

    return conversational_rag_chain


def get_ai_response(user_question):
    rag_chain = get_rag_chain()
    ai_response = rag_chain.stream(
        {
            "input": user_question
        },
        config={
            "configurable": {"session_id": "abc123"}
        },
    )

    return ai_response
