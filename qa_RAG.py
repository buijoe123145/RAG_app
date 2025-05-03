from langchain_community.llms import ctransformers
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceBgeEmbeddings


model_path = "models/vinallama-7b-chat_q5_0.gguf"
vector_db_path = "vectorstores/db_faiss"
embedding_model = HuggingFaceBgeEmbeddings()

def load_llm(model_path):
    llm = ctransformers.CTransformers(
        model=model_path,
        model_type="llama",
        temperature=0.1,
        max_new_tokens=1024,
    )
    return llm

def create_prompt(template):
    prompt = PromptTemplate(
        input_variables=["context","question"],
        template=template,
    )
    return prompt

def create_qa_chain(llm, prompt, db):
    llm_chain = RetrievalQA.from_chain_type(
        llm = llm,
        chain_type= "stuff",
        retriever = db.as_retriever(search_kwargs = {"k":2}, max_tokens_limit=512),
        return_source_documents = False,
        chain_type_kwargs= {'prompt': prompt}

    )
    return llm_chain

def read_db(vector_db_path, embedding_model):
    db = FAISS.load_local(
        vector_db_path,
        embedding_model,
        allow_dangerous_deserialization=True)
    return db

db = read_db(vector_db_path, embedding_model)
llm = load_llm(model_path)

template = """<|im_start|>system\nSử dụng thông tin sau đây để trả lời câu hỏi. Nếu bạn không biết câu trả lời, hãy nói không biết, đừng cố tạo ra câu trả lời\n
    {context}<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant"""
prompt = create_prompt(template)
chain = create_qa_chain(llm, prompt, db)

question = "Cần mấy năm làm việc trong lĩnh vực chứng khoán hoặc quản lý quỹ?"
response = chain({"query": question})
print(response)