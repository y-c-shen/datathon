import boto3
import streamlit as st
import os
import uuid
from langchain_community.embeddings import BedrockEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.llms.bedrock import Bedrock
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

from langchain_community.chat_models import BedrockChat

# S3 Client setup
s3_client = boto3.client(
    's3',
    aws_access_key_id='ASIAQNIHWFDGIVWHC73E',
    aws_secret_access_key='Ouz+CbnNyS5/kI3q+OVYHjrZaK0Fq62O4a3v530c',
    aws_session_token='IQoJb3JpZ2luX2VjEGIaCXVzLWVhc3QtMSJIMEYCIQDJF9FUymhPXDEjeG2przXMg2opXSpgrSYiXEkoXWVc5wIhANKvuDJjN/+gDStsV04Sss9ahRYNyxTaV7Jkx4kkyIboKqICCNv//////////wEQAhoMMDI4NDcwMjkwNjM2Igw9rOlWsHgM8AvaIcoq9gGPayXWrVaBZQCqEVX7I/lOTUD/g4S/Ox91e5SBSt77bxni6rb4sTo2dgZApAmJ/Fo/tJ9g4zp33+LLo5qgxWjoVBs6nCMhxYQuDG/0FX/u4AfaVw3j3GEFJqd2xr+XthosJ6af5tsY3iROyDEXVfQ2Wjj9eYrRBBR+EXGhpaoBOmW9OPpDlssWAKnvIpmOpX8ESX/R0Ct4DDnyJkikthtPwcdn9ALzHugoDSmbxmzROwe+XPVklSIZrefCNBRToYw1HTSHDVuHArmupjIJy+pq06cv9lp9xYLDJwvmLjBtJGTVSJRYHl9ED7O8qr5gyUcEhT5dlnEwkPGeuQY6nAENn+5Qu8pjsxFe2/jsvFIaE+hRVtqQi5xcOWRRGWDdhrBjCzq5z5bayJ6BvntHAfGwNhXsrzlo6MmA2L8aL3rJyjP48FA+bwwpWPk5r+Bn5Y+j+a2EIdnfcNdUBbfvbDSXcvvf5FNPuKgoX0jgIj3ipXdUWyJqnis+WsgyGgdiLCLP1T7XiCkKrCUvUUS6SKvJ28zpZZNTgE5vlgA=',
    region_name='us-west-2'
)

# Bedrock client setup
bedrock_client = boto3.client(
    service_name="bedrock-runtime",
    region_name="us-west-2",
    aws_access_key_id='ASIAQNIHWFDGIVWHC73E',
    aws_secret_access_key='Ouz+CbnNyS5/kI3q+OVYHjrZaK0Fq62O4a3v530c',
    aws_session_token='IQoJb3JpZ2luX2VjEGIaCXVzLWVhc3QtMSJIMEYCIQDJF9FUymhPXDEjeG2przXMg2opXSpgrSYiXEkoXWVc5wIhANKvuDJjN/+gDStsV04Sss9ahRYNyxTaV7Jkx4kkyIboKqICCNv//////////wEQAhoMMDI4NDcwMjkwNjM2Igw9rOlWsHgM8AvaIcoq9gGPayXWrVaBZQCqEVX7I/lOTUD/g4S/Ox91e5SBSt77bxni6rb4sTo2dgZApAmJ/Fo/tJ9g4zp33+LLo5qgxWjoVBs6nCMhxYQuDG/0FX/u4AfaVw3j3GEFJqd2xr+XthosJ6af5tsY3iROyDEXVfQ2Wjj9eYrRBBR+EXGhpaoBOmW9OPpDlssWAKnvIpmOpX8ESX/R0Ct4DDnyJkikthtPwcdn9ALzHugoDSmbxmzROwe+XPVklSIZrefCNBRToYw1HTSHDVuHArmupjIJy+pq06cv9lp9xYLDJwvmLjBtJGTVSJRYHl9ED7O8qr5gyUcEhT5dlnEwkPGeuQY6nAENn+5Qu8pjsxFe2/jsvFIaE+hRVtqQi5xcOWRRGWDdhrBjCzq5z5bayJ6BvntHAfGwNhXsrzlo6MmA2L8aL3rJyjP48FA+bwwpWPk5r+Bn5Y+j+a2EIdnfcNdUBbfvbDSXcvvf5FNPuKgoX0jgIj3ipXdUWyJqnis+WsgyGgdiLCLP1T7XiCkKrCUvUUS6SKvJ28zpZZNTgE5vlgA='
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
        return True
    except Exception as e:
        st.error(f"Error in create_vector_store: {str(e)}")
        return False

def load_index():
    s3_client.download_file(Bucket=BUCKET_NAME, Key="my_faiss.faiss", Filename=f"{folder_path}my_faiss.faiss")
    s3_client.download_file(Bucket=BUCKET_NAME, Key="my_faiss.pkl", Filename=f"{folder_path}my_faiss.pkl")

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
    st.title("Chat with PDF Demo - Admin Site")
    
    # File Upload Section
    st.header("Upload PDF")
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
                st.success(f"Created {len(splitted_docs)} document chunks")

            # Show sample chunks
            if st.checkbox("Show sample chunks"):
                st.write("Sample chunk 1:")
                st.write(splitted_docs[0])
                if len(splitted_docs) > 1:
                    st.write("Sample chunk 2:")
                    st.write(splitted_docs[1])

            # Create vector store
            with st.spinner("Creating vector store..."):
                result = create_vector_store(request_id, splitted_docs)
                if result:
                    st.success("PDF processed successfully!")
                else:
                    st.error("Error processing PDF. Please check the logs.")

            # Cleanup
            os.remove(saved_file_name)
            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

    # Query Section
    st.header("Query PDF")
    
    # Load index
    load_index()

    dir_list = os.listdir(folder_path)
    st.write(f"Files and Directories in {folder_path}")
    st.write(dir_list)

    # Create index
    faiss_index = FAISS.load_local(
        index_name="my_faiss",
        folder_path=folder_path,
        embeddings=bedrock_embeddings,
        allow_dangerous_deserialization=True
    )

    st.write("INDEX IS READY")
    question = st.text_input("Please ask your question")
    if st.button("Ask Question"):
        with st.spinner("Querying..."):
            llm = get_llm()
            response = get_response(llm, faiss_index, question)
            st.write(response)
            st.success("Done")

if __name__ == "__main__":
    main()