from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.prompts import PromptTemplate                         
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
# import ctransformers 
#from langchain.llms import CTransformers  in place of it, - pip install ctransformers
#here I am using retrieval answers only, for making it more chat history inclined we can 
# use other converstaional retrieval chains
from langchain.chains import RetrievalQA
import chainlit as cl

DB_FAISS_PATH = "vectorstores/db_faiss"

custom_prompt_template = """use the following pieces of information to generate the users questions. If you dont know the answers, 
do not build answers on your own
Context : {context}
Questions : {question}
only return the relatable and precise answers below.
Helpful Answer: 

""" 


def set_custom_prompt():
    """
    Prompt template for QA retrieval for each vector stores
    """
    prompt = PromptTemplate (template= custom_prompt_template, input_variables = ['context','question']) # original C:\Users\Hp\OneDrive\Desktop\AI\HF\llama
    return prompt

def load_llm():
    llm= CTransformers(
        model = "llama-2-7b-chat.ggmlv3.q8_0.bin", 
        model_type = "llama",
        max_new_token = 1024, 
        temprature = 0.5
    )
    return llm

def retrieval_qa_chain(llm, prompt, db):
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff", 
        retriever= db.as_retriever (search_kwargs ={'k':2}),
        return_source_documents =True,
        chain_type_kwargs={'prompt': prompt}
    )
    return qa_chain

def qa_bot():
    embeddings= HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                      model_kwargs={'device' : 'cpu'})
    db= FAISS.load_local(DB_FAISS_PATH, embeddings)
    llm= load_llm()
    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(llm, qa_prompt, db)

    return qa

#output function
def final_result(query):
    qa_result= qa_bot()
    response= qa_result({'query' : query})

    return response

################################################################################################
## chainlit functions
@cl.on_chat_start
async def start():
    chain = qa_bot()
    msg = cl.Message(content="Starting the bot...")
    await msg.send()
    msg.content = "Hi, Welcome..! This is a Facial Emotion Recognition System developed by Kishor Goswami. Ask anything you want to know about this project? 😀"
    await msg.update()

    cl.user_session.set("chain", chain)

@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("chain") 
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"]
    )
    cb.answer_reached = True
    res = await chain.acall(message.content, callbacks=[cb])
    answer = res["result"]
    sources = res["source_documents"]

    if sources:
        answer += f"\nSources:" + str(sources)
    else:
        answer += "\nNo sources found"

    await cl.Message(content=answer).send()