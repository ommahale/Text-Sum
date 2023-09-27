from transformers import pipeline
from transformers import BartForConditionalGeneration, AutoTokenizer
from fastapi import FastAPI
from pydantic import BaseModel

model = BartForConditionalGeneration.from_pretrained('./model/text-sum/')
tokenizer = AutoTokenizer.from_pretrained('./model/text-sum')
summ = pipeline('summarization', model=model, tokenizer=tokenizer)
app = FastAPI()

class Req(BaseModel):
    text: str

@app.post('/summarize/')
def getSummary(body:Req):
    return summ(body.text)