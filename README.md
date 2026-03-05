# Exam Learning Server API 說明文件

本伺服器提供基於強化學習 (Reinforcement Learning) 的考試預測與訓練介面。

## 基礎資訊
- **Base URL**: `http://127.0.0.1:8000`
- **內建機制**: 
  - **RL 模式**: 使用貝氏權重與 Softmax 進行探索學習。
  - **靜態模式 (Static Mode)**: 當連續 3 輪題目完全相同時觸發，改用「一次改一題」的策略鎖定正確答案。

---

## 1. 初始化伺服器
重置所有訓練權重、歷史紀錄並刪除本地存檔檔。

- **URL**: `/init`
- **Method**: `GET`
- **Response**:
  ```json
  {
    "message": "State reset successfully"
  }
  ```

---

## 2. 取得預測答案
發送目前的題目列表，伺服器會根據現有模型預測最佳選項。

- **URL**: `/predict`
- **Method**: `POST`
- **Request Body**:
  ```json
  {
    "questions": [
      {
        "question": "題目描述 1",
        "options": ["選項 A", "選項 B", "選項 C", "選項 D"] // 也可以是 [{"id": 1, "text": "A"}, ...] 或 {"A": "選項 A", ...}
      },
      ...
    ]
  }
  ```
- **Response**:
  ```json
  {
    "questions": [
      {
        "question": "題目描述 1",
        "options": ["選項 A", "選項 B", "選項 C", "選項 D"], // 保持與請求格式一致
        "chosen_index": 2
      },
      ...
    ],
    "static_mode": false
  }
  ```

---

## 3. 更新訓練資料
在完成考試後，將得分回傳給伺服器，用以修正模型。

- **URL**: `/update`
- **Method**: `POST`
- **Request Body**:
  ```json
  {
    "questions": [
      {
        "question": "題目描述 1",
        "options": ["選項 A", "選項 B", "選項 C", "選項 D"],
        "chosen_index": 2
      },
      ...
    ],
    "score": 8.0,
    "total_score": 10
  }
  ```
- **Response**:
  ```json
  {
    "message": "權重已更新",
    "static_mode": false
  }
  ```

---

## 5. Docker 部署
如果您習慣使用 Docker，可以使用以下指令快速啟動：

```bash
# 建立並啟動
docker-compose up --build -d
```

啟動後同樣可透過 `http://localhost:8000` 存取。

---

## 6. 自動存檔
伺服器會自動將權重儲存於根目錄的 `brain_data.json`。重新啟動伺服器後會自動載入。
