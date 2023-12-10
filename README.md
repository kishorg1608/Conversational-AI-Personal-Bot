[Conversational AI Personal Bot README.odt](https://github.com/kishorg1608/Conversational-AI-Personal-Bot/files/13628479/Conversational.AI.Personal.Bot.README.odt)

Conversational AI Personal Bot

## Overview

This project implements a Conversation AI model that utilizes the LLM (Large Language Model) Llama-2 to answer questions based on personal information extracted from PDF documents. The system leverages Langchain for document loading, embeddings, vector stores, and the CTransformers library for the Llama-2 model. The conversational aspect is implemented through the Chainlit library.

## Installation

To run this project, follow these steps:

1. Install dependencies:
    pip install langchain chainlit
2. Install the CTransformers library:
    pip install ctransformers
3. Download the LLM model and place it in the project directory.

4. Set up Faiss vector stores by running:
The Vector Store uses the PyPDFLoader to get the texts from the PDF file then splits it in chunks of 500. We further use the HuggingFaceEmbeddings model (model_name='sentence-transformers/all-MiniLM-L6-v2'). Using these embeddings and text chunks, we create a FAISS vector store locally in CPU.
Run the python file: 
    ```bash
    python ingest.py
    ```

## To use the model after making Vector Store :
1. Run the main script:
    ```bash
    Python AppModel.py
    ```
2. The chainlit system will give you a Question/Answer like UI to input questions based on a predefined template using personal information from PDFs.
3. The model will generate responses using the LLM Llama-2 and present relevant answers.
## Customization
- Adjust the `custom_prompt_template` in AppModel.py to modify the prompt used for generating questions.
- Customize the `DB_FAISS_PATH` variable for the location of the Faiss vector stores.

## Acknowledgements

- Langchain: [Langchain GitHub Repository](https://github.com/langchain/langchain)
- CTransformers: [CTransformers GitHub Repository](https://github.com/ctranz/ctranz)
- Chainlit: [Chainlit GitHub Repository](https://github.com/ctranz/chainlit)
- FAISS: [FAISS GitHub Repository](https://github.com/facebookresearch/faiss) 

## Feel free to explore and enhance the functionality of this Conversation AI model!
