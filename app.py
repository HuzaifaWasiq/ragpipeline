import streamlit as st
from backend import ensure_index, get_answer
st.set_page_config(page_title="Knowledge Chatbot", page_icon=":book:")

st.title("📚 Your Personal Knowledge Chatbot")

if "history" not in st.session_state:
    st.session_state.history = []

# Ensure index (builds on first run)
with st.spinner("Preparing the knowledge base (this may take a moment the first time)..."):
    ensure_index()

query = st.text_input("Ask something from your content:", key="input")

if st.button("Ask") or (query and st.session_state.get("auto_submit", False)):
    if query.strip() != "":
        with st.spinner("Thinking..."):
            answer, sources = get_answer(query)
        st.session_state.history.append({"q": query, "a": answer, "sources": sources})

# Display chat history (most recent first)
for item in reversed(st.session_state.history):
    st.markdown(f"**You:** {item['q']}")
    st.markdown(f"**Bot:** {item['a']}")
    # show sources (filenames)
    srcs = ', '.join(sorted(set([s.get('source','') for s in item['sources'] if s.get('source')]))) if item['sources'] else ""
    if srcs:
        st.caption(f"Sources: {srcs}")
    st.write("---")
