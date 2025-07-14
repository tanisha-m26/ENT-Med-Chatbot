from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain.text_splitter import RecursiveCharacterTextSplitter


from langchain_huggingface import HuggingFaceEmbeddings



#Extract Data from the PDF File
#Tabnine:Edit|Test|Explain|Document|Ask
def load_pdf_file(data):
    loader= DirectoryLoader(data,
                            glob="*.pdf",
                            loader_cls=PyPDFLoader)
    
    documents=loader.load()

    return documents


#Split the Data into Text Chunks
#Tabnine:Edit|Test|Explain|Document|Ask
def text_split(extracted_data):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    text_chunks=text_splitter.split_documents(extracted_data)

    return text_chunks


#Download the Embeddings from HuggingFace
#Tabnine:Edit|Test|Explain|Document|Ask
def download_hugging_face_embeddings():
    embeddings=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

    return embeddings


