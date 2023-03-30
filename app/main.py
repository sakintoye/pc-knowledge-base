# main.py

import os
from types import NoneType
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import json
import openai
from langchain.llms import AzureOpenAI
from llama_index.langchain_helpers.text_splitter import TokenTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from llama_index import Document, LangchainEmbedding, QueryMode
from llama_index import (
    GPTListIndex,
    GPTSimpleVectorIndex,
    SimpleDirectoryReader, 
    LLMPredictor,
    PromptHelper,
    SimpleWebPageReader,
    BeautifulSoupWebReader
)
import time

openai.api_type = ""
openai.api_base = ""
openai.api_version = "2022-12-01"
os.environ["OPENAI_API_KEY"] = ""
openai.api_key = ""

llm = AzureOpenAI(deployment_name="davinci-003", model_kwargs={
"api_key": openai.api_key,
"api_base": openai.api_base,
"api_type": openai.api_type,
"api_version": openai.api_version,
})
llm_predictor = LLMPredictor(llm=llm)

documents = BeautifulSoupWebReader().load_data([
    "https://learn.microsoft.com/en-us/partner-center/reconciliation-faq",
    "https://learn.microsoft.com/en-us/partner-center/read-your-bill",
    "https://learn.microsoft.com/en-us/partner-center/billing-basics",
    "https://learn.microsoft.com/en-us/partner-center/understand-your-invoice",
    "https://learn.microsoft.com/en-us/partner-center/use-the-reconciliation-files",
    "https://learn.microsoft.com/en-us/partner-center/license-based-recon-files",
    "https://learn.microsoft.com/en-us/partner-center/usage-based-recon-files",
    "https://learn.microsoft.com/en-us/partner-center/modern-invoice-reconciliation-file",
    "https://learn.microsoft.com/en-us/partner-center/daily-rated-usage-recon-files",
    "https://learn.microsoft.com/en-us/partner-center/azure-credit-offer-balance",
    "https://learn.microsoft.com/en-us/partner-center/recon-file-charge-types",
    "https://learn.microsoft.com/en-us/partner-center/set-an-azure-spending-budget-for-your-customers",
    "https://learn.microsoft.com/en-us/partner-center/common-billing-scenarios",
    "https://learn.microsoft.com/en-us/partner-center/common-billing-scenarios-monthly",
    "https://learn.microsoft.com/en-us/partner-center/common-billing-scenarios-annual",
    "https://learn.microsoft.com/en-us/partner-center/billing-frequency-changes",
    "https://learn.microsoft.com/en-us/partner-center/common-billing-scenarios-onetime-recurring",
    "https://learn.microsoft.com/en-us/partner-center/azure-plan-billing",
    "https://learn.microsoft.com/en-us/partner-center/azure-savings",
    "https://learn.microsoft.com/en-us/partner-center/csp-commercial-marketplace-billing",
    "https://learn.microsoft.com/en-us/partner-center/common-billing-scenarios-saas",
    "https://learn.microsoft.com/en-us/partner-center/provide-billing-support",
    "https://learn.microsoft.com/en-us/partner-center/partner-earned-credit",
    "https://learn.microsoft.com/en-us/partner-center/partner-earned-credit-explanation",
    "https://learn.microsoft.com/en-us/partner-center/azure-roles-perms-pec",
    "https://learn.microsoft.com/en-us/partner-center/partner-earned-credit-faq",
    "https://learn.microsoft.com/en-us/partner-center/partner-earned-credit-troubleshoot",
    "https://learn.microsoft.com/en-us/partner-center/organization-tax-info",
    "https://learn.microsoft.com/en-us/partner-center/tax-and-tax-exemptions",
    "https://learn.microsoft.com/en-us/partner-center/tax-and-fees-voice-services",
    "https://learn.microsoft.com/en-us/partner-center/request-credit",
    "https://learn.microsoft.com/en-us/partner-center/withholding-tax-credit-form"
])

embedding_llm = LangchainEmbedding(OpenAIEmbeddings(
document_model_name="text-similarity-babbage-001", # must be a model that supports embeddings
query_model_name="text-similarity-babbage-001"
))

# max LLM token input size
max_input_size = 500
# set number of output tokens
num_output = 200
# set maximum chunk overlap
max_chunk_overlap = 20

prompt_helper = PromptHelper(max_input_size, num_output, max_chunk_overlap)
# index = GPTSimpleVectorIndex(documents)
# index = GPTSimpleVectorIndex(documents, embed_model=embedding_llm, llm_predictor=llm_predictor, prompt_helper=prompt_helper)


index = GPTSimpleVectorIndex([], embed_model=embedding_llm, llm_predictor=llm_predictor, prompt_helper=prompt_helper)

for doc in documents:
    index.insert(doc)
    # ideally ingestion should not invoke any embedding model but it does and we result in throttling
    # so self throttling with a fixed time
    time.sleep(5)


class Ask(BaseModel):
    query: str

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Use the POST endpoint to make your query!"}

@app.post("/")
async def root(ask: Ask):

    query = ask.query
    answer = index.query(query, mode="embedding", verbose=True)

    print(answer.get_formatted_sources())
    source_url = answer.source_nodes[0].extra_info['URL']
    
    return {"message": answer.response, "url": source_url, "diagnostics": answer.source_nodes}