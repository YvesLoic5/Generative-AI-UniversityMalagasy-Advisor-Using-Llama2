from src.helper import load_pdf, text_split, download_hugging_face_embeddings
import pinecone
from dotenv import load_dotenv
import os
from langchain_pinecone import Pinecone as PineconeStore

load_dotenv()
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')


extracted_data = load_pdf("/home/yvesloic/Documents/informatique/Défi/text-generative-chatbot-using-llama2/data")
text_chunks = text_split(extracted_data)
embeddings = download_hugging_face_embeddings()

#Initializing the Pinecone
os.environ['PINECONE_API_KEY'] = PINECONE_API_KEY

#Creating Embeddings for Each of The Text Chunks & storing
vector_database_index = PineconeStore.from_documents(
                                            index_name = "university-bot", 
                                            documents = text_chunks, 
                                            embedding = embeddings)