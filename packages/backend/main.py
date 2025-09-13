from fastapi import FastAPI

app = FastAPI()

@app.post("/qa_session")
async def start_qa_session(request: Request):
