# Phase 2: Document Upload + Text Extraction

# Step 1:Imports
import streamlit as st
import pandas as pd
from pypdf import PdfReader
import tempfile
import os

# Step 2: Page Configuration
st.set_page_config(
    page_title="AI Business Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# step 3: Load CSS
def load_css():
    try:
        with open("assets/style.css") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        pass  # Handle the case where the CSS file is not found

load_css()

# Step 4: Session State Initialization
defaults = {
    "documents": [],
    "doc_count": 0,
    "total_chars": 0,
    "vectorstore": None,
    "chunks": [],
    "page": "upload",
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

# Helper function to extract text from PDF
def extract_pdf(file):
    """ Extract text from a PDF file using pypdf."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
        f.write(file.read())
        path = f.name

    reader = PdfReader(path)
    text = " "
    for page in reader.pages:
        t = page.extract_text()
        if t:
            text += t + "\n"
    return text

def extract_excel(file):
    """ Extract text from Excel """
    df = pd.read_excel(file)
    return df.to_string()

def extract_csv(file):
    """ Extract text from CSV """
    df = pd.read_csv(file)
    return df.to_string()

# Step 6: Hero Section
st.markdown("""
<div class="hero">
    <h1>📊 AI Business Intelligence</h1>
    <p>Upload your business documents and get AI-powered insights instantly</p>
</div>
""", unsafe_allow_html=True
)

# Upload Section
st.markdown("## 📄 Upload Documents")

col1, col2 = st.columns([2, 1])

with col1:
    uploaded_files = st.file_uploader(
        "Choose file (pdf, eXCEL, csv)",
        type=["pdf", "xlsx", "xls", "csv"],
        accept_multiple_files=True,
        help="Upload Multiple Documnents at Once"
    )

with col2:
    st.markdown("""
    <div class="info-box">
        <h4>💡 Supported Formats</h4>
        <ul>
            <li>📕 PDF documents</li>
            <li>📗 Excel spreadsheets</li>
            <li>📘 CSV data files</li>
        </ul>
    </div>
    """, unsafe_allow_html=True
    )

# Process Uploaded Files
if uploaded_files:
    documents = []

    with st.spinner(f"Processing {len(uploaded_files)} file(s)..."):
        for uploaded_file in uploaded_files:
            file_name = uploaded_file.name
            file_ext = file_name.split(".")[-1].lower()

            try:
                # Extract text based on file type
                if file_ext == "pdf":
                    text = extract_pdf(uploaded_file)
                elif file_ext in ["xlsx", "xls"]:
                    text = extract_excel(uploaded_file)
                elif file_ext == "csv":
                    text = extract_csv(uploaded_file)
                else:
                    continue  # Skip unsupported file types

                documents.append({
                    "name" : file_name,
                    "type" : file_ext,
                    "text" : text,
                    "size" : len(text)
                })

            except Exception as e:
                st.error(f"❌ Error processing {file_name}: {e}")

        # Save to session state
        st.session_state.documents = documents
        st.session_state.doc_count = len(documents)
        st.session_state.total_chars = sum(d["size"] for d in documents)

        # Success message
        st.success(f"✅ Successfully processed {len(documents)} document(s)")

# Step 9: Display stats
if st.session_state.doc_count > 0:
    st.markdown("---")
    st.markdown("## 📊 Document Statistics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value">{st.session_state.doc_count}</div>
            <div class="stat-label">Documents</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value">{st.session_state.total_chars:,}</div>
            <div class="stat-label">Total Characters</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        avg = st.session_state.total_chars // st.session_state.doc_count
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value">{avg:,}</div>
            <div class="stat-label">Avg Characters</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Step 10: Document list
    st.markdown("---")
    st.markdown("## 📋 Uploaded Documents")
    
    for i, doc in enumerate(st.session_state.documents):
        with st.expander(f"📄 {doc['name']} ({doc['size']:,} chars)"):
            st.markdown(f"**Type:** {doc['type'].upper()}")
            st.markdown(f"**Size:** {doc['size']:,} characters")
            st.markdown("**Preview:**")
            st.text(doc["text"][:1000] + "...")
    
    # Step 11: Next steps
    st.markdown("---")
    st.info("🎯.....................")

else:
    st.info("👆 Please upload at least one document to get started")

################################################################
################################################################
################################################################
################################################################

# PHASE 3: RAG SETUP (Embeddings + Vector Store)

# Step 1: Langchain imports
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
import os

# step 2 : RAG setup Function 
def setup_rag(documents):
    """ Break documents into chunks, embed them, and create a vector store."""

    # 2.1:  Convert the all Document into Object
    docs = []
    for doc in documents:
        docs.append(Document(
            page_content=doc["text"],
            metadata={"source": doc["name"], "type": doc["type"]}
        ))
    
    # 2.2: Break Text into Chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    
    chunks = splitter.split_documents(docs)
    
    # 2.3: Made Embeddings model  (free, local)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    # 2.4: Made Vector store 
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    
    return chunks, vectorstore

# If Documents Loaded SO do the RAG setup
if st.session_state.doc_count > 0 and st.session_state.vectorstore is None:
    with st.spinner("Setting up RAG (Chunk + Embaddings)...."):
        chunks, vectorstore = setup_rag(st.session_state.documents)
        st.session_state.chunks = chunks
        st.session_state.vectorstore = vectorstore
    st.success(f"✅ RAG ready! Created {len(chunks)} chunks.")


# RAG setup Display 
if st.session_state.vectorstore is not None:
    st.markdown("---")
    st.markdown("## 🔍 RAG Status")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value">{len(st.session_state.chunks)}</div>
            <div class="stat-label">Chunks</div>
        </div>
        """, unsafe_allow_html=True
        )

    with col2:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value">384</div>
            <div class="stat-label">Dimensions</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value">✅</div>
            <div class="stat-label">Vector Store</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Next steps
    st.markdown("---")
    st.info("🎯 **Next Phase:** Question & Answering.. ")

##################################
#################################
################################

# PHASE 4: Q&A FEATURE (RAG Chain)

from langchain_openai import ChatOpenAI
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

# Step 1: Load the Grog API Key
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY"))

# Groq LLM setup (OpenAI-compatible)
llm = ChatOpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=GROQ_API_KEY,
    model="openai/gpt-oss-20b",
    temperature=0.3,
    max_tokens=2000
)


# Custom prompt for Q&A
qa_prompt_template = """You are a helpful AI Business Intelligence Analyst.

Use the following context from the uploaded documents to answer the question.
If you don't know the answer based on the context, say "I couldn't find this in the documents."
Do NOT make up information.

Context:
{context}

Question: {question}

Answer in a clear, professional way:"""

QA_PROMPT = PromptTemplate(
    template=qa_prompt_template,
    input_variables=["context", "question"]
)

# Create Retriever if vector store is ready
if st.session_state.vectorstore is not None:
    retriever = st.session_state.vectorstore.as_retriever(
        search_kwargs={"k" : 3}
    )

    # Creat RAG chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_PROMPT}
    )

    # Q&A UI
    st.markdown("---")
    st.markdown("## 💬 Ask Questions About Your Documents")
    
    question = st.text_input(
        "Your question:",
        placeholder="e.g., What were the sales trends last quarter?",
        key="qa_question"
    )
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        ask_button = st.button("🔍 Get Answer", use_container_width=True)


    # Genrate Answer if button clicked
    if ask_button and question:
        with st.spinner("Searching your documents..."):
            try:
                result = qa_chain.invoke({"query" : question})
                answer = result["result"]
                sources = result["source_documents"]

                # Display Answer
                
                st.markdown("### 📝 Answer")
                st.markdown(f"""
                <div class="answer-box">
                    {answer}
                </div>
                """, unsafe_allow_html=True)
                
                # Display sources
                with st.expander("📚 View Sources"):
                    for i, doc in enumerate(sources):
                        st.markdown(f"**Source {i+1}:** {doc.metadata.get('source', 'Unknown')}")
                        st.text(doc.page_content[:300] + "...")
                        st.markdown("---")
                
            except Exception as e:
                st.error(f"❌ Error: {e}")
    
    elif ask_button and not question:
        st.warning("⚠️ Please type a question first")

else:
    st.info("👆 Please upload documents first to enable Q&A")



# ============================================
# ============================================
# ============================================
# ============================================

# PHASE 5: ANALYSIS AGENT (Insights Generation)

# Analysis Function 
def analyze_documents(documents, analysis_type="general"):
    """ Gnerate Insight From Documents """

    # Combine All Documents Text
    combined_text = "\n\n".join([doc["text"] for doc in documents])

    # Create prompts based on analysis points.
    if analysis_type == "general":
        prompt = f""" You are a Business Intellugence Analyst
        
Analyze the following business documents and provide:

1. **Key Findings** (5-7 bullet points)
2. **Trends & Patterns** (3-5 points)
3. **Risks & Concerns** (3-5 points)
4. **Opportunities** (3-5 points)
5. **Recommendations** (3-5 actionable points)

Documents:
{combined_text[:8000]}

Analysis:"""
    
    elif analysis_type == "financial":
        prompt = f"""You are a Financial Analyst.

Analyze the following financial documents and provide:

1. **Revenue Trends**
2. **Cost Analysis**
3. **Profitability Insights**
4. **Financial Risks**
5. **Recommendations**

Documents:
{combined_text[:8000]}

Financial Analysis:"""
    
    elif analysis_type == "sales":
        prompt = f"""You are a Sales Analyst.

Analyze the following sales documents and provide:

1. **Sales Performance**
2. **Top Products/Services**
3. **Customer Trends**
4. **Areas of Improvement**
5. **Recommendations**

Documents:
{combined_text[:8000]}

Sales Analysis:"""
    
    # Analysis From LLM 
    try:
        response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {"role": "system", "content": "You are a Business Intelligence Analyst. Provide clear, actionable insights."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=4000,
            temperature=0.3
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {e}" 


# Step 2: Analysis UI
if st.session_state.doc_count > 0:
    st.markdown("---")
    st.markdown("## 🔍 Auto Analysis")
    
    # Analysis type selector
    col1, col2 = st.columns([2, 1])
    
    with col1:
        analysis_type = st.selectbox(
            "Choose analysis type:",
            ["general", "financial", "sales"],
            format_func=lambda x: {
                "general": "📊 General Business Analysis",
                "financial": "💰 Financial Analysis",
                "sales": "📈 Sales Analysis"
            }[x]
        )
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        analyze_button = st.button("🔍 Generate Insights", use_container_width=True)
    
    # Step 3: Analysis generate karo
    if analyze_button:
        with st.spinner("AI analyzing your documents..."):
            insights = analyze_documents(st.session_state.documents, analysis_type)
            
            # Display insights
            st.markdown("### 📊 Insights")
            st.markdown(f"""
            <div class="insights-box">
                {insights}
            </div>
            """, unsafe_allow_html=True)
            
            # Save to session state
            st.session_state.insights = insights
            
            # Download button
            st.download_button(
                "📥 Download Insights",
                data=insights,
                file_name=f"insights_{analysis_type}.txt",
                mime="text/plain"
            )
    
    # Show previous insights if exists
    if "insights" in st.session_state and st.session_state.insights:
        with st.expander("📋 Previous Insights"):
            st.markdown(st.session_state.insights)



# ============================================
# PHASE 6: REPORT GENERATOR
# ============================================

# Step 1: Report generation function
def generate_report(documents, insights, analysis_type="general"):
    """Professional report generate karo"""
    
    # creat the  Documents summary
    doc_summary = "\n".join([
        f"- {doc['name']} ({doc['type'].upper()}, {doc['size']:,} chars)"
        for doc in documents
    ])
    
    # Report prompt creat
    prompt = f"""You are a Senior Business Intelligence Analyst.

Generate a PROFESSIONAL BUSINESS REPORT based on the following:

**Documents Analyzed:**
{doc_summary}

**Key Insights:**
{insights}

**Report Structure:**
1. **Executive Summary** (3-4 lines)
2. **Documents Overview** (list of documents)
3. **Key Findings** (detailed analysis)
4. **Trends & Patterns**
5. **Risks & Concerns**
6. **Opportunities**
7. **Recommendations** (actionable steps)
8. **Conclusion**

Make it professional, clear, and actionable.
Use proper headings and formatting.

Generate the full report:"""
    
    # LLM se report lo
    try:
        response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {"role": "system", "content": "You are a Senior Business Intelligence Analyst. Write professional reports."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=4000,
            temperature=0.3
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {e}"

# Step 2: Report UI
if st.session_state.doc_count > 0 and "insights" in st.session_state and st.session_state.insights:
    st.markdown("---")
    st.markdown("## 📋 Generate Business Report")
    
    st.markdown("""
    <div class="info-box">
        <h4>📄 Report Contents</h4>
        <ul>
            <li>Executive Summary</li>
            <li>Documents Overview</li>
            <li>Key Findings</li>
            <li>Trends & Patterns</li>
            <li>Risks & Concerns</li>
            <li>Opportunities</li>
            <li>Recommendations</li>
            <li>Conclusion</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        generate_report_btn = st.button("📊 Generate Report", use_container_width=True)
    
    # Step 3: Generat Report 
    if generate_report_btn:
        with st.spinner("Generating professional report..."):
            report = generate_report(
                st.session_state.documents,
                st.session_state.insights,
                "general"
            )
            
            # Save to session state
            st.session_state.report = report
            
            st.success("✅ Report generated!")
    
    # Step 4: Display report
    if "report" in st.session_state and st.session_state.report:
        st.markdown("---")
        st.markdown("### 📄 Generated Report")
        
        st.markdown(f"""
        <div class="report-box">
            {st.session_state.report}
        </div>
        """, unsafe_allow_html=True)
        
        # Download buttons
        col1, col2 = st.columns(2)
        
        with col1:
            st.download_button(
                "📥 Download Report (TXT)",
                data=st.session_state.report,
                file_name="business_report.txt",
                mime="text/plain",
                use_container_width=True
            )
        
        with col2:
            # Markdown download
            st.download_button(
                "📥 Download Report (MD)",
                data=st.session_state.report,
                file_name="business_report.md",
                mime="text/markdown",
                use_container_width=True
            )

else:
    if st.session_state.doc_count == 0:
        st.info("👆 Upload documents first to generate reports")
    elif "insights" not in st.session_state or not st.session_state.insights:
        st.info("👆 Generate insights first (Phase 5) to create reports")



# ============================================
# PHASE 7: DASHBOARD + CHARTS
# ============================================

# Step 1: Imports
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

# Step 2: Dashboard UI
if st.session_state.doc_count > 0:
    st.markdown("---")
    st.markdown("## 📊 Business Dashboard")
    
    # Step 3: KPI Cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-icon">📄</div>
            <div class="kpi-value">{st.session_state.doc_count}</div>
            <div class="kpi-label">Documents</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        total_chars = st.session_state.total_chars
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-icon">📝</div>
            <div class="kpi-value">{total_chars:,}</div>
            <div class="kpi-label">Characters</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        chunks = len(st.session_state.chunks) if st.session_state.chunks else 0
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-icon">🔍</div>
            <div class="kpi-value">{chunks}</div>
            <div class="kpi-label">Chunks</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        insights_status = "✅" if "insights" in st.session_state and st.session_state.insights else "⏳"
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-icon">💡</div>
            <div class="kpi-value">{insights_status}</div>
            <div class="kpi-label">Insights</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Step 4: Charts Section
    st.markdown("### 📈 Document Analytics")
    
    col1, col2 = st.columns(2)
    
    # Chart 1: Document Size Distribution (Pie Chart)
    with col1:
        st.markdown("#### 📄 Document Sizes")
        
        doc_data = pd.DataFrame([
            {"name": doc["name"][:20], "size": doc["size"]}
            for doc in st.session_state.documents
        ])
        
        if not doc_data.empty:
            fig_pie = px.pie(
                doc_data,
                values="size",
                names="name",
                color_discrete_sequence=px.colors.sequential.Purples_r,
                hole=0.4
            )
            fig_pie.update_layout(
                showlegend=True,
                height=350,
                margin=dict(l=20, r=20, t=20, b=20)
            )
            st.plotly_chart(fig_pie, use_container_width=True)
    
    # Chart 2: Character Count (Bar Chart)
    with col2:
        st.markdown("#### 📊 Character Count")
        
        if not doc_data.empty:
            fig_bar = px.bar(
                doc_data,
                x="name",
                y="size",
                color="size",
                color_continuous_scale="Purples"
            )
            fig_bar.update_layout(
                height=350,
                margin=dict(l=20, r=20, t=20, b=20),
                xaxis_title="",
                yaxis_title="Characters"
            )
            st.plotly_chart(fig_bar, use_container_width=True)
    
    # Step 5: Document Type Distribution
    st.markdown("### 📁 Document Types")
    
    type_data = pd.DataFrame([
        {"type": doc["type"].upper(), "count": 1}
        for doc in st.session_state.documents
    ]).groupby("type").count().reset_index()
    
    if not type_data.empty:
        fig_type = px.bar(
            type_data,
            x="type",
            y="count",
            color="type",
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig_type.update_layout(
            height=300,
            margin=dict(l=20, r=20, t=20, b=20),
            xaxis_title="Document Type",
            yaxis_title="Count",
            showlegend=False
        )
        st.plotly_chart(fig_type, use_container_width=True)
    
    # Step 6: Insights Summary
    if "insights" in st.session_state and st.session_state.insights:
        st.markdown("---")
        st.markdown("### 💡 Key Insights Summary")
        
        st.markdown(f"""
        <div class="insights-box">
            {st.session_state.insights[:1000]}...
        </div>
        """, unsafe_allow_html=True)
    
    # Step 7: Export Dashboard
    st.markdown("---")
    st.markdown("### 📥 Export Dashboard Data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Download document stats
        stats_df = pd.DataFrame([
            {
                "Document": doc["name"],
                "Type": doc["type"],
                "Characters": doc["size"]
            }
            for doc in st.session_state.documents
        ])
        
        csv = stats_df.to_csv(index=False)
        st.download_button(
            "📥 Download Stats (CSV)",
            data=csv,
            file_name="document_stats.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col2:
        # Download insights
        if "insights" in st.session_state and st.session_state.insights:
            st.download_button(
                "📥 Download Insights (TXT)",
                data=st.session_state.insights,
                file_name="insights.txt",
                mime="text/plain",
                use_container_width=True
            )

else:
    st.info("👆 Upload documents first to see the dashboard")




# ============================================
# PHASE 8: UI POLISH
# ============================================

# Step 1: Page config with better settings
st.set_page_config(
    page_title="AI Business Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"  # ← Sidebar expanded
)

# Step 2: Sidebar with navigation
with st.sidebar:
    st.markdown("""
    <div class="sidebar-header">
        <h2>📊 AI BI Analyst</h2>
        <p>Business Intelligence & Document Analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Navigation
    st.markdown("### 🧭 Navigation")
    page = st.radio(
        "Go to:",
        ["📄 Upload Documents", "💬 Q&A", "🔍 Analysis", "📋 Report", "📊 Dashboard"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    # Status
    st.markdown("### 📊 Status")
    st.markdown(f"""
    <div class="status-item">
        <span>Documents:</span>
        <strong>{st.session_state.doc_count}</strong>
    </div>
    """, unsafe_allow_html=True)
    
    if st.session_state.vectorstore is not None:
        st.markdown(f"""
        <div class="status-item">
            <span>Chunks:</span>
            <strong>{len(st.session_state.chunks)}</strong>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Reset button
    if st.button("🔄 Reset App", use_container_width=True):
        for key in defaults:
            st.session_state[key] = defaults[key]
        st.rerun()
    
    st.markdown("---")
    st.markdown("""
    <div class="sidebar-footer">
        <p>Built by <strong>Asif Nawaz</strong></p>
        <p>AI/ML Engineer</p>
    </div>
    """, unsafe_allow_html=True)

# Step 3: Page routing
if page == "📄 Upload Documents":
    # Existing upload code
    pass
elif page == "💬 Q&A":
    # Existing Q&A code
    pass
elif page == "🔍 Analysis":
    # Existing analysis code
    pass
elif page == "📋 Report":
    # Existing report code
    pass
elif page == "📊 Dashboard":
    # Existing dashboard code
    pass