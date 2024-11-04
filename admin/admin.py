import boto3
import streamlit as st
import os
import uuid
from langchain_community.embeddings import BedrockEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.chat_models import BedrockChat

# S3 Client setup
s3_client = boto3.client(
    's3',
    aws_access_key_id='YOUR_ACCESS_KEY_ID',
    aws_secret_access_key='YOUR_SECRET_ACCESS_KEY',
    aws_session_token='YOUR_SESSION_TOKEN',
    region_name='YOUR_REGION'
)

# Bedrock client setup
bedrock_client = boto3.client(
    service_name="bedrock-runtime",
    region_name='YOUR_REGION',
    aws_access_key_id='YOUR_ACCESS_KEY_ID',
    aws_secret_access_key='YOUR_SECRET_ACCESS_KEY',
    aws_session_token='YOUR_SESSION_TOKEN'
    )

# Initialize embeddings
bedrock_embeddings = BedrockEmbeddings(
    model_id="amazon.titan-embed-text-v1",
    client=bedrock_client
)

# Set bucket name correctly
BUCKET_NAME = "rapport-annuel"
folder_path = "/tmp/"

def get_unique_id():
    return str(uuid.uuid4())

def split_text(pages, chunk_size, chunk_overlap):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(pages)
    return docs

def create_vector_store(request_id, documents):
    try:
        vectorstore_faiss = FAISS.from_documents(documents, bedrock_embeddings)
        file_name = f"{request_id}.bin"
        
        # Ensure the directory exists
        os.makedirs(folder_path, exist_ok=True)
        
        vectorstore_faiss.save_local(folder_path=folder_path, index_name=file_name)

        # Upload to S3
        s3_client.upload_file(
            Filename=os.path.join(folder_path, f"{file_name}.faiss"),
            Bucket=BUCKET_NAME,
            Key="my_faiss.faiss"
        )
        s3_client.upload_file(
            Filename=os.path.join(folder_path, f"{file_name}.pkl"),
            Bucket=BUCKET_NAME,
            Key="my_faiss.pkl"
        )
        
        # Store the file information in session state
        if 'uploaded_files' not in st.session_state:
            st.session_state.uploaded_files = []
        st.session_state.uploaded_files.append({
            'id': request_id,
            'name': file_name,
            'faiss_file': f"{file_name}.faiss",
            'pkl_file': f"{file_name}.pkl"
        })
        
        return True
    except Exception as e:
        st.error(f"Error in create_vector_store: {str(e)}")
        return False

def load_vector_store():
    """Load the vector store from S3 and return the FAISS index"""
    try:
        # Download files from S3
        s3_client.download_file(Bucket=BUCKET_NAME, Key="my_faiss.faiss", Filename=f"{folder_path}my_faiss.faiss")
        s3_client.download_file(Bucket=BUCKET_NAME, Key="my_faiss.pkl", Filename=f"{folder_path}my_faiss.pkl")
        
        # Load the FAISS index
        faiss_index = FAISS.load_local(
            index_name="my_faiss",
            folder_path=folder_path,
            embeddings=bedrock_embeddings,
            allow_dangerous_deserialization=True
        )
        return faiss_index
    except Exception as e:
        st.error(f"Error loading vector store: {str(e)}")
        return None

def get_llm():
    llm = BedrockChat(
        model_id="anthropic.claude-3-sonnet-20240229-v1:0",
        client=bedrock_client,
        model_kwargs={
            "max_tokens": 512,
            "temperature": 0.7,
            "top_p": 1,
            "top_k": 250,
            "anthropic_version": "bedrock-2023-05-31"
        }
    )
    return llm

def get_response(llm, vectorstore, question):
    prompt_template = """
    Human: Use the following context to answer the question. If you don't know the answer, just say that you don't know, don't try to make up an answer.

    Context: {context}

    Question: {question}

    Assistant: I'll do my best to answer based on the context provided.

    Human: Great, please provide your answer now. Be concise and be confident in your answer. 

    Assistant: """

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(
            search_type="similarity", search_kwargs={"k": 5}
        ),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    answer = qa({"query": question})
    return answer['result']

def main():
    # Initialize session state
    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = []
    if 'vector_store_loaded' not in st.session_state:
        st.session_state.vector_store_loaded = False
    if 'faiss_index' not in st.session_state:
        st.session_state.faiss_index = None
        
    st.title("Welcome to SummariX !")
    
    # File Upload Section
    st.header("Upload PDF to start")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    if uploaded_file is not None:
        request_id = get_unique_id()
        st.info(f"Request ID: {request_id}")
        
        # Save uploaded file
        saved_file_name = f"{request_id}.pdf"
        with open(saved_file_name, mode="wb") as w:
            w.write(uploaded_file.getvalue())

        try:
            # Load PDF
            with st.spinner("Loading PDF..."):
                loader = PyPDFLoader(saved_file_name)
                pages = loader.load_and_split()
                st.success(f"Successfully loaded {len(pages)} pages")

            # Split text
            with st.spinner("Processing document..."):
                splitted_docs = split_text(pages, 1000, 200)
                #st.success(f"Created {len(splitted_docs)} document chunks")

            # Show sample chunks
            if st.checkbox("Show sample chunks"):
                st.write("Sample chunk 1:")
                st.write(splitted_docs[0])
                if len(splitted_docs) > 1:
                    st.write("Sample chunk 2:")
                    st.write(splitted_docs[1])

            # Create and load vector store if not already loaded
            if not st.session_state.vector_store_loaded:
                with st.spinner("Creating vector store..."):
                    result = create_vector_store(request_id, splitted_docs)
                    if result:
                        st.success("PDF processed successfully!")
                        # Load the vector store immediately after creation
                        st.session_state.faiss_index = load_vector_store()
                        st.session_state.vector_store_loaded = True
                    else:
                        st.error("Error processing PDF. Please check the logs.")

            # Cleanup
            os.remove(saved_file_name)
            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

    # Query Section
    #st.header("Query PDF")

    if st.session_state.vector_store_loaded:
        # Add Quick Summary button
        if st.button("📄 Quick Summary"):
            with st.spinner("Generating summary..."):
                llm = get_llm()
                summary = get_response(llm, st.session_state.faiss_index, "Summarize the main concepts of the file. Have a confident, but concise tone")
                st.markdown("### Document Summary")
                st.write(summary)
                st.markdown("---")
        question = st.text_input("Please ask your question")
        if st.button("Ask Question"):
            with st.spinner("Querying..."):
                llm = get_llm()
                response = get_response(llm, st.session_state.faiss_index, question)
                st.write(response)
                #st.success("Done")
    else:
        st.info("Please upload a PDF file first to start querying.")

if __name__ == "__main__":
    main()