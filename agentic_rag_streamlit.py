import os
import base64
import urllib.parse
from fpdf import FPDF
from dotenv import load_dotenv
import streamlit as st

# MUST be the first Streamlit call
st.set_page_config(page_title="Agentic RAG Chatbot", page_icon="🦜")

# OCR
from PIL import Image
import pytesseract

# LangChain Core
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import SupabaseVectorStore

# Supabase
from supabase.client import create_client, Client

# --- Load environment variables ---
load_dotenv()
supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_SERVICE_KEY")

supabase: Client = create_client(supabase_url, supabase_key)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

vector_store = SupabaseVectorStore(
    embedding=embeddings,
    client=supabase,
    table_name="documents",
    query_name="match_documents",
)

llm = ChatOpenAI(model="gpt-4o", temperature=0)

# --- PDF generation utility ---
def create_pdf(text: str) -> bytes:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    for line in text.split("\n"):
        try:
            pdf.multi_cell(0, 10, txt=line)
        except Exception:
            pdf.multi_cell(0, 10, txt="[Encoding Error: Skipped line]")
    return bytes(pdf.output(dest="S"))

# --- OCR from uploaded image ---
def extract_text_from_image(image_file):
    image = Image.open(image_file)
    return pytesseract.image_to_string(image)

# --- Upload and Ingest Document ---
st.sidebar.markdown("## 📤 Upload Document")
uploaded_image = st.sidebar.file_uploader("Upload scanned contract/image", type=["jpg", "jpeg", "png"])

if uploaded_image:
    extracted_text = extract_text_from_image(uploaded_image)
    vector_store.add_texts([extracted_text], metadatas=[{"source": uploaded_image.name}])
    st.sidebar.success("✅ Document processed and indexed!")

# --- Custom Retrieval Tool with References ---
@tool(response_format="content_and_artifact")
def retrieve(query: str):
    """Retrieve legal documents or references relevant to the query."""
    retrieved_docs = vector_store.similarity_search(query, k=3)
    if not retrieved_docs:
        return "No relevant documents found.", []

    content_blocks = []
    references = []

    for i, doc in enumerate(retrieved_docs, start=1):
        marker = f"[{i}]"
        content_blocks.append(f"{marker} {doc.page_content.strip()}")

        meta = doc.metadata
        source = meta.get("source") or meta.get("title") or "Unnamed Document"
        precedent = meta.get("precedent", "")
        link = meta.get("link")

        ref_line = f"{marker} Source: **{source}**"
        if precedent:
            ref_line += f" | Precedent: *{precedent}*"
        if link:
            ref_line += f" ([link]({link}))"
        references.append(ref_line)

    content = "\n\n".join(content_blocks)
    refs = "\n".join(references)

    markdown_output = f"""### 📘 Retrieved Information\n\n{content}

---

### 📎 References
{refs}
"""
    return markdown_output, retrieved_docs

# --- Tools ---
tools = [retrieve]

# --- Prompt with Required Vars ---
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a legal assistant chatbot helping users by answering queries using retrieved legal documents.
Use tools when necessary and always cite using [1], [2] style.
Show references clearly in a 'References' section."""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
    ("placeholder", "{agent_scratchpad}")
])

agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# --- Streamlit UI ---
st.title("🦜 Legal Agentic RAG Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_sessions" not in st.session_state:
    st.session_state.chat_sessions = {}

chat_title = st.text_input("📁 Name this Chat", value="Untitled Chat")

# --- Sidebar History ---
with st.sidebar:
    st.markdown("## 🗂 Chat History")
    for title, logs in st.session_state.chat_sessions.items():
        with st.expander(title):
            for i, msg in enumerate(logs):
                st.markdown(f"**Q{i+1}:** {msg['user']}")
                st.markdown(f"**A{i+1}:** {msg['bot']}")

# --- Show Chat History ---
for message in st.session_state.messages:
    role = "user" if isinstance(message, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(message.content)

# --- Input Field ---
user_question = st.chat_input("Ask a legal question...")

if user_question:
    with st.chat_message("user"):
        st.markdown(user_question)
    st.session_state.messages.append(HumanMessage(user_question))

    with st.spinner("🔍 Searching..."):
        result = agent_executor.invoke({
            "input": user_question,
            "chat_history": st.session_state.messages,
            "agent_scratchpad": [],
            "tools": tools
        })

    ai_message = result["output"]

    with st.chat_message("assistant"):
        st.markdown(ai_message, unsafe_allow_html=False)

        pdf_bytes = create_pdf(ai_message)
        b64_pdf = base64.b64encode(pdf_bytes).decode()
        pdf_link = f'<a href="data:application/pdf;base64,{b64_pdf}" download="chat_response.pdf">📄 Download as PDF</a>'
        st.markdown(pdf_link, unsafe_allow_html=True)

        wa_text = urllib.parse.quote(f"Legal AI Bot Response:\n\n{ai_message}")
        wa_link = f"https://wa.me/?text={wa_text}"
        st.markdown(f"[💬 Share via WhatsApp]({wa_link})", unsafe_allow_html=True)

    st.session_state.messages.append(AIMessage(ai_message))

    if chat_title not in st.session_state.chat_sessions:
        st.session_state.chat_sessions[chat_title] = []
    st.session_state.chat_sessions[chat_title].append({
        "user": user_question,
        "bot": ai_message
    })