import numpy as np
import json
import os
from typing import List, Dict, Tuple

class ExamLearner:
    def __init__(self, storage_path: str = "data/brain_data.json"):
        self.storage_path = storage_path
        self.lr = 0.1  # 學習率調整，配合歸一化邏輯
        self.temperature = 1.0  # 初始溫度，用於控制探索比例
        self.weights = {}  # 題目文本對應權重向量的映射
        self.history_weight_count = 0 
        self.stable_rounds = 0
        self.static_mode = False
        
        # 爬山法 (Hill Climbing) 靜態模式狀態
        self.best_static_score = -1
        self.best_static_choices = {} # 題目文本 -> 目前最佳選擇索引
        self.hc_current_q_idx = 0      # 目前正在測試的題目索引
        self.hc_current_choice_offset = 0 # 目前正在嘗試的選項偏移量
        self.last_test_info = None    # 紀錄上一輪測試了哪一題
        
        self.last_predicted = [] # 上一次預測的紀錄 (題目文本, 選擇索引, 是否在測試中)
        self.load()

    def load(self):
        if os.path.exists(self.storage_path):
            with open(self.storage_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.weights = {k: np.array(v) for k, v in data.get("weights", {}).items()}
                self.history_weight_count = data.get("history_weight_count", 0)
                self.stable_rounds = data.get("stable_rounds", 0)
                self.static_mode = data.get("static_mode", False)
                self.best_static_score = data.get("best_static_score", -1)
                self.best_static_choices = data.get("best_static_choices", {})
                self.hc_current_q_idx = data.get("hc_current_q_idx", 0)
                self.hc_current_choice_offset = data.get("hc_current_choice_offset", 0)

    def save(self):
        data = {
            "weights": {k: v.tolist() for k, v in self.weights.items()},
            "history_weight_count": self.history_weight_count,
            "stable_rounds": self.stable_rounds,
            "static_mode": self.static_mode,
            "best_static_score": self.best_static_score,
            "best_static_choices": self.best_static_choices,
            "hc_current_q_idx": self.hc_current_q_idx,
            "hc_current_choice_offset": self.hc_current_choice_offset
        }
        with open(self.storage_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def reset(self):
        try:
            if os.path.exists(self.storage_path):
                os.remove(self.storage_path)
        except Exception as e:
            print(f"無法刪除資料檔 (可能被鎖定): {e}")
            # 即使刪除失敗，也嘗試清空內容
            try:
                with open(self.storage_path, "w", encoding="utf-8") as f:
                    json.dump({}, f)
            except:
                pass
        
        self.weights = {}
        self.history_weight_count = 0
        self.stable_rounds = 0
        self.static_mode = False
        self.best_static_score = -1
        self.best_static_choices = {}
        self.hc_current_q_idx = 0
        self.hc_current_choice_offset = 0
        self.last_predicted = []
        print("[!] 系統狀態已重置")

    def softmax(self, x, temperature=1.0):
        # 避免溫度過低導致計算溢出
        temp = max(temperature, 0.01)
        x_scaled = x / temp
        e = np.exp(x_scaled - np.max(x_scaled))
        return e / e.sum()

    def predict(self, questions: List[Dict]) -> List[int]:
        # 更新權重穩定度計數
        prev_weight_count = self.history_weight_count
        current_weight_count = len(self.weights)
        
        if current_weight_count > 0 and current_weight_count == prev_weight_count:
            self.stable_rounds += 1
        else:
            self.stable_rounds = 0
        
        self.history_weight_count = current_weight_count

        # 靜態模式觸發檢查
        if current_weight_count == len(questions) and self.stable_rounds >= 3:
            if not self.static_mode:
                print("[*] 符合靜態規定，啟動爬山法精確搜尋模式...")
                self.static_mode = True
                # 初始化最佳答案為當前權重最大的選項
                for q_text, w in self.weights.items():
                    if q_text not in self.best_static_choices:
                        self.best_static_choices[q_text] = int(np.argmax(w))
                self.hc_current_q_idx = 0
                self.hc_current_choice_offset = 0
                self.best_static_score = -1

        results = []
        self.last_predicted = []
        self.last_test_info = None
        

        for i, q in enumerate(questions):
            q_text = q["question"]
            num_options = len(q["options"])
            
            if q_text not in self.weights:
                self.weights[q_text] = np.ones(num_options)
            
            is_testing_this_q = False
            
            if self.static_mode:
                # 取得目前此題的最佳索引
                base_idx = self.best_static_choices.get(q_text, 0)
                
                # 如果 offset > 0，代表目前正在搜尋中（尚未達到滿分或搜尋完畢）
                # 且這輪輪到測試這一題
                if self.hc_current_choice_offset > 0 and i == (self.hc_current_q_idx % len(questions)):
                    choice = (base_idx + self.hc_current_choice_offset) % num_options
                    is_testing_this_q = True
                    self.last_test_info = {
                        "question": q_text,
                        "base_idx": base_idx,
                        "test_idx": choice,
                        "num_options": num_options
                    }
                else:
                    choice = base_idx
            else:
                # RL 模式：根據穩定輪數動態降低溫度，漸漸轉向開發 (Exploitation)
                current_temp = max(0.1, self.temperature / (1 + self.stable_rounds * 0.1))
                probs = self.softmax(self.weights[q_text], temperature=current_temp)
                choice = int(np.random.choice(range(num_options), p=probs))
            
            results.append(choice)
            self.last_predicted.append({
                "question": q_text, 
                "choice_index": choice, 
                "is_testing": is_testing_this_q
            })
            
        return results

    def update(self, score: float, total_score: int, chosen_data: List[Dict]):
        if not chosen_data:
            return

        if self.static_mode:
            # --- 爬山法 (Hill Climbing) 更新邏輯 ---
            
            if self.best_static_score == -1:
                self.best_static_score = score
                # 如果首場即滿分（score == total_score），鎖定不再測試
                self.hc_current_choice_offset = 0 if score >= total_score else 1
                print(f"  [#] 基準分數已建立: {score}")
                if score >= total_score:
                    print("  [!] 首場預測即獲得滿分，鎖定解答")
                self.save()
                return

            # 如果已經獲取過滿分，且本輪沒測試資訊，直接跳過訓練
            if self.best_static_score >= total_score and not self.last_test_info:
                return

            # 2. 評估測試結果
            if self.last_test_info:
                q_text = self.last_test_info["question"]
                test_idx = self.last_test_info["test_idx"]
                base_idx = self.last_test_info["base_idx"]
                num_options = self.last_test_info["num_options"]
                
                if score > self.best_static_score:
                    # 分數提高：測試的選項是對的
                    print(f"  [+] 進步! {q_text[:15]}...: {base_idx} -> {test_idx} ({self.best_static_score} -> {score})")
                    self.best_static_choices[q_text] = test_idx
                    self.best_static_score = score
                    
                    # 強化權重
                    self.weights[q_text] = np.zeros(num_options)
                    self.weights[q_text][test_idx] = 20.0
                    
                    # 如果達到滿分，鎖定 offset 為 0，停止探索
                    if score >= total_score:
                        print(f"  [!] 已達到滿分 ({score}/{total_score})，終止爬山法測試並鎖定解答")
                        self.hc_current_choice_offset = 0
                    else:
                        # 換下一題測試
                        self.hc_current_q_idx += 1
                        self.hc_current_choice_offset = 1
                elif score < self.best_static_score:
                    # 分數降低：測試的選項是錯的，原來的較好 (或是選錯了)
                    print(f"  [-] 退步! {q_text[:15]}...: 恢復 {base_idx}")
                    # 懲罰測試的選項
                    self.weights[q_text][test_idx] = -10.0
                    # 既然分數降低，原本的 base_idx 更有可能是正確的
                    self.hc_current_q_idx += 1
                    
                    # 如果基準已經是滿分（可能因為分數規模變動），恢復後應停止測試
                    # 這裡使用一個較寬鬆的判斷或依賴之前的 total_score 邏輯
                    if self.best_static_score >= total_score:
                        self.hc_current_choice_offset = 0
                    else:
                        self.hc_current_choice_offset = 1
                else:
                    # 分數沒變：原本與測試的可能都錯，或此題不影響
                    self.hc_current_choice_offset += 1
                    if self.hc_current_choice_offset >= num_options:
                        print(f"  [!] 題目 {q_text[:15]}... 所有選項皆測試完畢")
                        self.hc_current_q_idx += 1
                        self.hc_current_choice_offset = 1
            else:
                # 沒在測試（可能是剛手動改了什麼），維持現狀
                self.hc_current_choice_offset = 1
        else:
            # --- 優化後的強化學習 (RL) 更新 ---
            # 1. 動態計算期望值：根據該次測驗所有題目的選項數算出「亂猜的平均分」
            dynamic_expected = sum(1.0 / item["num_options"] for item in chosen_data)
            
            # 2. 計算差異並進行題數歸一化，確保更新幅度不隨題目數量劇烈波動
            raw_delta = score - dynamic_expected
            normalized_delta = raw_delta / max(1, total_score)
            
            for item in chosen_data:
                q_text = item["question"]
                num_options = item["num_options"]
                choice = item["chosen_index"]
                
                if q_text not in self.weights:
                    self.weights[q_text] = np.ones(num_options)
                
                if choice < len(self.weights[q_text]):
                    # 強化選中的選項，弱化其他選項
                    # 更新值 = 學習率 * 歸一化後的表現差異
                    update_val = self.lr * normalized_delta
                    self.weights[q_text][choice] += update_val
                    
                    # 3. 權重衰減 (Weight Decay)：輕微的正則化，避免權重過大並保持調整彈性
                    self.weights[q_text] *= 0.99
        
        self.save()
