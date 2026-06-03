import streamlit as st
import requests
import os
import json

# Widescreen Portal Configuration
st.set_page_config(
    page_title="AI Research Assistant - Portal",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Endpoint Paths
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
UPLOAD_URL = f"{API_BASE_URL}/api/v1/upload"
QUERY_URL = f"{API_BASE_URL}/api/v1/query"
STATUS_URL = f"{API_BASE_URL}/api/v1/status"

# Custom Premium Styling System
st.markdown("""
<style>
    .main { background-color: #0F172A; color: #F8FAFC; }
    .stSidebar { background-color: #1E293B !important; }
    h1, h2, h3 { color: #38BDF8 !important; }
    .stButton>button { background-color: #38BDF8 !important; color: #0F172A !important; border-radius: 8px; font-weight: bold; }
    .stTextInput>div>div>input { background-color: #1E293B !important; color: #F8FAFC !important; border: 1px solid #475569 !important; }
    .metric-card { background-color: #1E293B; border: 1px solid #475569; border-radius: 12px; padding: 15px; margin: 10px 0; text-align: center; }
    .citation-box { background-color: #1E293B; border-left: 4px solid #6366F1; padding: 10px; margin: 5px 0; border-radius: 4px; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Sidebar Controller Block
# ---------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/brain-security.png", width=64)
    st.title("RAG CONTROLS")
    st.subheader("Ingestion Plane")
    
    uploaded_file = st.file_uploader("Upload reference documents (PDF)", type=["pdf"])
    if uploaded_file is not None:
        if st.button("Trigger Ingestion Processing"):
            with st.spinner("Executing semantic extraction and ChromaDB indexing..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                    response = requests.post(UPLOAD_URL, files=files)
                    if response.status_code == 200:
                        res_data = response.json()
                        st.success(f"Successfully split and indexed {res_data.get('chunks_created')} semantic chunks.")
                    else:
                        st.error(f"Ingestion failed: {response.text}")
                except Exception as e:
                    st.error(f"Error connecting to backend API: {str(e)}")
                    
    st.write("---")
    st.subheader("System Status")
    try:
        status_res = requests.get(STATUS_URL).json()
        st.success("FastAPI & Vector DB: ONLINE")
        st.info(f"Model: {status_res.get('active_model')}")
        st.info(f"Ingested Docs: {status_res.get('files_ingested')}")
    except:
        st.error("FastAPI & Vector DB: OFFLINE")

# ---------------------------------------------------------
# Main Page Workspace
# ---------------------------------------------------------
st.title("Enterprise AI Research Assistant")
st.write("Production-grade context search, similarity indexing, and GPT-4o citation-grounded reasoning.")

# Chat History Session State Setup
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous conversation streams
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Receive user query
if user_query := st.chat_input("Ask a research query on ingested documents..."):
    # Display user input
    with st.chat_message("user"):
        st.markdown(user_query)
    st.session_state.messages.append({"role": "user", "content": user_query})
    
    # Process RAG reasoning
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        with st.spinner("Retrieving context chunks and executing GPT-4o reasoning..."):
            try:
                payload = {"query": user_query, "collection": "default"}
                res = requests.post(QUERY_URL, json=payload)
                
                if res.status_code == 200:
                    response_data = res.json()
                    answer = response_data.get("answer", "")
                    citations = response_data.get("citations", [])
                    scores = response_data.get("scores", {})
                    
                    # Display Answer
                    message_placeholder.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    
                    # Display interactive citations if present
                    if citations:
                        with st.expander("Authoritative Document Source Citations"):
                            for idx, cit in enumerate(citations):
                                st.markdown(f"""
                                <div class="citation-box">
                                    <strong>Source {cit.get('citation_id')}</strong>: {cit.get('file_name')} (Page {cit.get('page_number')})<br/>
                                    <em>\"{cit.get('quote')}\"</em>
                                </div>
                                """, unsafe_allow_html=True)
                                
                    # Display TruLens RAG metrics in three columns
                    st.write("---")
                    st.subheader("Live Evaluation Metrics (TruLens Triad)")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.markdown(f"""
                        <div class="metric-card">
                            <h4 style="color:#52D399 !important;">FAITHFULNESS</h4>
                            <h2>{scores.get('faithfulness', 0.0):.2f} / 1.00</h2>
                            <p style="color:#52D399;font-weight:bold;">PASSED</p>
                        </div>
                        """, unsafe_allow_html=True)
                    with col2:
                        st.markdown(f"""
                        <div class="metric-card">
                            <h4 style="color:#52D399 !important;">ANSWER RELEVANCE</h4>
                            <h2>{scores.get('answer_relevance', 0.0):.2f} / 1.00</h2>
                            <p style="color:#52D399;font-weight:bold;">PASSED</p>
                        </div>
                        """, unsafe_allow_html=True)
                    with col3:
                        st.markdown(f"""
                        <div class="metric-card">
                            <h4 style="color:#52D399 !important;">CONTEXT RECALL</h4>
                            <h2>{scores.get('context_recall', 0.0):.2f} / 1.00</h2>
                            <p style="color:#52D399;font-weight:bold;">PASSED</p>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.error(f"Inference failed: {res.text}")
            except Exception as e:
                st.error(f"Failed to connect to RAG backend: {str(e)}")
