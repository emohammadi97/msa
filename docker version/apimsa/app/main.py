import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from .msa import MSA
from pydantic import BaseModel

# app = FastAPI(root_path=os.environ.get('ROOT_PATH','/'))
app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Item(BaseModel):
    sents: list

@app.post('/')
def main(item: Item):

    return MSA(item.sents,gap_penalty=0.49,gap_weight=1,scoring_fn=None).df


# uvicorn main:app --reload
# python -m uvicorn main:app --reload