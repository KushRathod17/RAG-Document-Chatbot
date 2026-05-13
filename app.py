import streamlit as st
from ingest import load_pdf, split_into_chunks, create_vector_store
from rag_pipeline import load_vector_store, create_rag_chain

st.set_page_config(page_title="RAG Chatbot", page_icon="📄")
st.title("📄 RAG Document Chatbot")

# Session state
if "chain" not in st.session_state:
    st.session_state.chain = None
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar for PDF upload
with st.sidebar:
    st.header("📂 Upload Document")
    uploaded_file = st.file_uploader("Upload your PDF", type="pdf")
    
    if uploaded_file is not None:
        if st.button("Process PDF", type="primary"):
            with st.spinner("Indexing..."):
                with open("data/temp.pdf", "wb") as f:
                    f.write(uploaded_file.getvalue())
                pages = load_pdf("data/temp.pdf")
                chunks = split_into_chunks(pages)
                vector_store = create_vector_store(chunks)
                chain, retriever = create_rag_chain(vector_store)
                st.session_state.chain = chain
                st.session_state.retriever = retriever
                st.session_state.messages = []
                st.success(f"✅ Indexed {len(chunks)} chunks!")

    if st.session_state.chain:
        st.info("✅ Document ready!")
    else:
        st.warning("⚠️ Upload a PDF to start")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question about your document..."):
    if st.session_state.chain is None:
        st.error("Please upload and process a PDF first!")
    else:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        # Stream assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = st.session_state.chain.invoke(prompt)
                docs = st.session_state.retriever.invoke(prompt)
            
            st.write(answer)
            
            # Show sources
            with st.expander("📄 Sources"):
                for doc in docs:
                    st.caption(f"Page {doc.metadata['page']}: {doc.page_content[:150]}...")

        # Save assistant message
        st.session_state.messages.append({"role": "assistant", "content": answer})