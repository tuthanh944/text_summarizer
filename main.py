from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

app = FastAPI()
templates = Jinja2Templates(directory="templates")

tokenizer = AutoTokenizer.from_pretrained("ntkhoi/mt5-vi-news-summarization")
model = AutoModelForSeq2SeqLM.from_pretrained("ntkhoi/mt5-vi-news-summarization")

@app.get("/", response_class=HTMLResponse)
async def get_form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "summary": None})

@app.post("/", response_class=HTMLResponse)
async def summarize(request: Request, text: str = Form(...)):
    input_ids = tokenizer(text, return_tensors="pt", truncation=True).input_ids
    output_ids = model.generate(input_ids, max_length=100, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return templates.TemplateResponse("index.html", {"request": request, "summary": summary, "text": text})
