import httpx
import random
import math
import numpy as np

# ===== 符合使用者參考資料的參數 =====
NUM_QUESTIONS = 30
CHOICES = 4
DRAW_PER_EXAM = 10
ROUNDS = 200

# 建立固定的題目池
QUESTIONS_POOL = [
    {
        "question": f"Question {i}?",
        "options": [f"Option {j}" for j in range(CHOICES)],
        "answer": random.randint(0, CHOICES-1)
    }
    for i in range(NUM_QUESTIONS)
]

SERVER_URL = "http://127.0.0.1:8000"

def test_simulation():
    try:
        # 1. 重置狀態
        print("正在初始化伺服器...")
        httpx.get(f"{SERVER_URL}/init")
        
        scores = []
        
        for r in range(ROUNDS):
            # Draw random questions
            batch = random.sample(QUESTIONS_POOL, DRAW_PER_EXAM)
            req_batch = [{"question": q["question"], "options": q["options"]} for q in batch]
            
            # 2. 預測答案
            resp = httpx.post(f"{SERVER_URL}/predict", json={"questions": req_batch})
            resp_data = resp.json()
            preds = resp_data["questions"]
            static_mode = resp_data["static_mode"]
            
            # 3. 計算分數
            score = 0
            for i, p in enumerate(preds):
                if p["chosen_index"] == batch[i]["answer"]:
                    score += 1
            
            # 4. 更新模型
            httpx.post(f"{SERVER_URL}/update", json={
                "questions": preds,
                "score": float(score), 
                "total_score": DRAW_PER_EXAM
            })
            
            scores.append(score)
            if (r+1) % math.ceil(ROUNDS/10) == 0:
                print(f"第 {r+1} 輪: 分數 {score}/{DRAW_PER_EXAM}, 靜態模式: {static_mode}")
        
        print("\n--- 第二階段: 靜態題目集 ---")
        # 為了符合「總題數等於測驗題數」的靜態規定，我們從池中選出固定數量的題目作為「全部題目」
        static_pool = random.sample(QUESTIONS_POOL, DRAW_PER_EXAM)
        
        # 先重置伺服器以進入乾淨的測試環境 (或僅為了確保題數匹配)
        httpx.get(f"{SERVER_URL}/init")
        
        for r in range(ROUNDS):
             req_batch = [{"question": q["question"], "options": q["options"]} for q in static_pool]
             resp = httpx.post(f"{SERVER_URL}/predict", json={"questions": req_batch})
             resp_data = resp.json()
             preds = resp_data["questions"]
             
             score = sum(1 for i, p in enumerate(preds) if p["chosen_index"] == static_pool[i]["answer"])
             
             httpx.post(f"{SERVER_URL}/update", json={
                 "questions": preds,
                 "score": float(score), 
                 "total_score": DRAW_PER_EXAM
             })
             print(f"靜態考輪 {r+1}: 分數 {score}/{DRAW_PER_EXAM}, 靜態模式: {resp_data['static_mode']}")

        print("模擬測試完成。")

    except Exception as e:
        print(f"Test failed: {e}")

if __name__ == "__main__":
    test_simulation()
