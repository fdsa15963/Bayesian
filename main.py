from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
from brain import ExamLearner
import uvicorn

app = FastAPI(title="Exam Learning Server")
learner = ExamLearner()

@app.get("/")
async def read_root():
    return {
        "message": "Exam Learning Server API",
        "endpoints": {
            "/init": {
                "method": "GET", 
                "desc": "Reset state",
                "response": {"message": "string"}
            },
            "/predict": {
                "method": "POST",
                "desc": "Get predictions",
                "request": {
                    "questions": [{"question": "string", "options": ["string"]}]
                },
                "response": {
                    "questions": [{"question": "string", "options": ["string"], "chosen_index": "int"}]
                }
            },
            "/update": {
                "method": "POST",
                "desc": "Update weights",
                "request": {
                    "questions": [{"question": "string", "options": ["string"], "chosen_index": "int"}],
                    "score": "float",
                    "total_score": "int"
                },
                "response": {"message": "string"}
            }
        }
    }

class Question(BaseModel):
    question: str
    options: List[str]

class PredictRequest(BaseModel):
    questions: List[Question]

class QuestionWithChoice(BaseModel):
    question: str
    options: List[str]
    chosen_index: int

class UpdateRequest(BaseModel):
    questions: List[QuestionWithChoice]
    score: float
    total_score: int

@app.get("/init")
async def init_state():
    learner.reset()
    return {"message": "狀態已成功重置"}

@app.post("/predict")
async def predict_answers(req: PredictRequest):
    # 將 Pydantic 模型轉換為字典供 learner 使用
    q_dicts = [q.model_dump() for q in req.questions]
    indices = learner.predict(q_dicts)
    
    # 同時回傳題目、選項與所選索引的列表
    results = []
    for i, q in enumerate(req.questions):
        idx = indices[i]
        results.append({
            "question": q.question,
            "options": q.options,
            "chosen_index": idx
        })
    
    # 外層包裝 questions，並附帶 static_mode 狀態方便前端判斷
    return {
        "questions": results,
        "static_mode": learner.static_mode
    }

@app.post("/update")
async def update_model(req: UpdateRequest):
    # 將選擇的資料轉換為字典格式供 learner 使用
    data = [
        {
            "question": q.question, 
            "num_options": len(q.options), 
            "chosen_index": q.chosen_index
        } 
        for q in req.questions
    ]
    learner.update(req.score, req.total_score, data)
    return {"message": "權重已更新"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
