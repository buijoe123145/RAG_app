from langchain_community.llms import ctransformers
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

model_path = "models/vinallama-7b-chat_q5_0.gguf"

def load_llm(model_path):
    llm = ctransformers.CTransformers(
        model=model_path,
        model_type="llama",
        temperature=0.1,
        max_new_tokens=512,
    )
    return llm

def crete_prompt(template):
    prompt = PromptTemplate(
        input_variables=["question"],
        template=template,
    )
    return prompt

def create_simple_chain(llm, prompt):
    chain = LLMChain(
        llm=llm,
        prompt=prompt,
    )
    return chain

template = """<|im_start|>system
Bạn là một trợ lí AI hữu ích. Hãy trả lời người dùng một cách chính xác.
<|im_end|>
<|im_start|>user
{question}<|im_end|>
<|im_start|>assistant"""

prompt = crete_prompt(template)
llm = load_llm(model_path)
chain = create_simple_chain(llm, prompt)

question = "1 cộng 1 bằng mấy?"
response = chain.invoke({"question": question})
print(response)