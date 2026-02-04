import os
import streamlit as st
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda



load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

if not HF_TOKEN:
    st.error("HUGGINGFACEHUB_API_TOKEN not found in .env")
    st.stop()



st.set_page_config(page_title="PadCare GPT", page_icon="🌱")
st.title("🌱 PadCare GPT")
st.caption("Answers are generated using publicly available PadCare sources only.")


@st.cache_resource
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return FAISS.load_local(
        "vectorstore",
        embeddings,
        allow_dangerous_deserialization=True
    )

vectorstore = load_vectorstore()
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})


hf_client = InferenceClient(
    model="HuggingFaceH4/zephyr-7b-beta",
    token=HF_TOKEN
)


def hf_generate(prompt) -> str:
    # Convert ChatPromptValue -> string
    if hasattr(prompt, "to_string"):
        prompt_text = prompt.to_string()
    else:
        prompt_text = str(prompt)

    response = hf_client.chat_completion(
        messages=[
            {
                "role": "user",
                "content": prompt_text
            }
        ],
        max_tokens=512,
        temperature=0.0,
    )

    return response.choices[0].message["content"]




llm = RunnableLambda(hf_generate)

def format_answer_as_markdown(text: str) -> str:
    lines = text.split("\n")
    formatted = []

    for line in lines:
        line = line.strip()

        # Convert common patterns into markdown
        if line.lower().startswith(("problem", "solution", "process", "impact", "summary", "investor", "overview")):
            formatted.append(f"### {line}")

        elif line.startswith(("-", "•")):
            formatted.append(f"- {line.lstrip('-• ').strip()}")

        elif line.endswith(":"):
            formatted.append(f"**{line}**")

        else:
            formatted.append(line)

    return "\n".join(formatted)


prompt = ChatPromptTemplate.from_template(
    """
You are PadCare GPT, an internal knowledge assistant.

RULES:
- Use ONLY the provided context
- Keep answers factual and concise
- Use clear sections when appropriate
- Prefer bullet points for lists
- Do NOT add external knowledge
- If information is missing, say:
  "I don’t have that information from public PadCare sources."

CONTEXT:
{context}

QUESTION:
{question}

FORMAT:
- Use headings where helpful
- Use bullet points for clarity

ANSWER:
"""
)




chain = (
    {
        "context": retriever,
        "question": lambda x: x
    }
    | prompt
    | llm
    | StrOutputParser()
)


query = st.text_input("Ask something about PadCare:")

if query:
    with st.spinner("Thinking..."):
        answer = chain.invoke(query)
        docs = retriever.invoke(query)

    st.subheader("Answer")
    formatted_answer = format_answer_as_markdown(answer)
    st.markdown(formatted_answer)

    st.subheader("Sources")
    sources = {doc.metadata.get("source", "Unknown source") for doc in docs}
    for source in sources:
        st.write(f"- {source}")
