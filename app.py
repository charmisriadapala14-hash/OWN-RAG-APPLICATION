import streamlit as st
import PyPDF2
import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
page_title="Own RAG Application",
page_icon="🧠"
)

st.title("🧠 Own RAG Application")
st.write("Upload PDF documents and ask questions using AI.")

api_key = st.sidebar.text_input(
"OpenRouter API Key",
type="password"
)

uploaded_files = st.file_uploader(
"Upload PDF Documents",
type=["pdf"],
accept_multiple_files=True
)

question = st.text_input("Ask a question about your documents")

@st.cache_resource
def load_embedding_model():
return SentenceTransformer("all-MiniLM-L6-v2")

def extract_pdf_text(pdf_file):
reader = PyPDF2.PdfReader(pdf_file)
text = ""

for page in reader.pages:
    text += page.extract_text() or ""

return text

def split_text(text, chunk_size=500):
return [
text[i:i + chunk_size]
for i in range(0, len(text), chunk_size)
if text[i:i + chunk_size].strip()
]

def retrieve_documents(chunks, question):
model = load_embedding_model()

document_embeddings = model.encode(chunks)
question_embedding = model.encode([question])

similarities = cosine_similarity(
    question_embedding,
    document_embeddings
)[0]

top_indices = np.argsort(similarities)[-3:][::-1]

return [chunks[i] for i in top_indices]

if st.button("Generate Answer"):

if not api_key or not uploaded_files or not question.strip():
    st.warning("Enter API key, upload PDFs and ask a question.")

else:
    try:
        all_text = ""

        for file in uploaded_files:
            all_text += extract_pdf_text(file) + "\n"

        chunks = split_text(all_text)

        if not chunks:
            st.error("No readable text found in the uploaded PDFs.")
        else:
            with st.spinner("Retrieving relevant information..."):

                relevant_chunks = retrieve_documents(
                    chunks, question
                )

                context = "\n\n".join(relevant_chunks)

            client = OpenAI(
                api_key=api_key,
                base_url="https://openrouter.ai/api/v1"
            )

            with st.spinner("Generating AI answer..."):

                response = client.chat.completions.create(
                    model="openai/gpt-4o-mini",
                    messages=[
                        {
                            "role": "system",
                            "content": "Answer using only the supplied document context. If the answer is unavailable, say so."
                        },
                        {
                            "role": "user",
                            "content": f"Context:\n{context}\n\nQuestion: {question}"
                        }
                    ]
                )

            st.subheader("💡 AI Answer")
            st.write(
                response.choices[0].message.content
            )

            with st.expander("Retrieved Document Context"):
                st.write(context)

    except Exception as e:
        st.error(f"An error occurred: {e}")
