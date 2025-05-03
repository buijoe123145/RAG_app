# RAG_app

## 1. Install all the packages
You need to install all the packages in the setup.txt file in order to run the application appropriately.
```
pip install -r setup.txt
```
---
## 2. Download the LLM from huggingface

Link to LLM

**I don't have GPU so I use GGUF LLM so that I can run the app local on my CPU**
```
https://huggingface.co/vilm/vinallama-7b-chat-GGUF
```

After download, you can put the LLM file in the "models" folder if you dont want to change the code.

---
## 3. Prepare pdf file to make a vector database
You can use any file of any content that you want to. Make sure that you run the `prep_vector_db.py` to create a vector database for later retrieval.

## Test the LLM (Optinal)
You can run `testl_lm_model.py` file to test the LLM using some basic prompt.

## 4. Ask the LLM for some relative information from PDF file
At this point you cant run the `qa_RAG.py` file to ask LLM some information from PDF file. (Change the question in this `qa_RAG.py` file)

