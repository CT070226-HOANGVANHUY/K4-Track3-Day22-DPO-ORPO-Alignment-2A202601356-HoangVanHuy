# Colab runbook — T4 + XAH custom judge

## 1. Trước khi mở Colab

1. Thu hồi API key từng xuất hiện trong ảnh/chat và tạo key mới.
2. Push toàn bộ code chuẩn bị lên nhánh `main` của repo này.
3. Mở `colab/Lab22_DPO_T4.ipynb` từ badge trong README, chọn T4 GPU.
4. Trong Colab → Secrets, thêm `XAH_API_KEY`; không paste key vào cell.

## 2. Chạy và dừng ở các checkpoint cần phán đoán

1. Chạy setup và hai preflight. Không tiếp tục nếu 401/403; nếu gateway không
   chạy được, NB4 vẫn có thể dùng manual-only nhưng NB6/β win-rate chưa hoàn tất.
2. Chạy NB1–NB3. Kiểm tra loss/reward curves trước khi tiếp tục.
3. Chạy API cell NB4 một lần. Ở cell **5a**, đọc `side_by_side.jsonl`, thay đủ
   tám `TODO` bằng `sft`, `dpo`, hoặc `tie` cùng lý do, rồi chỉ rerun từ cell 5a.
4. Chạy β-sweep. Mở
   `/content/lab22/data/eval/beta_sweep_results.json`, điền manual audit cho
   prompt 1, 5, 8 ở mỗi β, lưu lại rồi gọi `backup_stage('beta-sweep')` lần nữa.
5. Chạy NB6. Ở cell **5a**, thay đủ 10 manual-audit `TODO`; không chạy lại 100
   API calls. Kiểm tra `alpaca_lite_judgments.json` đã chứa manual verdicts.
6. Chạy NB5 cuối. Chụp cell load Q4_K_M và phản hồi tiếng Việt thành
   `06-gguf-smoke.png`, sau đó gọi `backup_stage('NB5', include_large=True)`.

## 3. Hoàn thiện evidence

- Tạo đúng tám ảnh được liệt kê trong `submission/screenshots/README.md`.
- Điền Reflection bằng số liệu thật; hoàn tất §3, §6, §7 ≥150 từ và §5 ≥100 từ.
- Với judge, ghi đúng: `XAH OpenAI-compatible custom judge
  (rouyea98/gpt-5.4) + manual validation`.
- Chạy `make verify-full` trong `/content/lab22`. Các lỗi còn lại là checklist
  cần hoàn tất, không được thay bằng số liệu giả.

## 4. Đưa kết quả về repo

1. Tải notebook đang mở bằng **File → Download → Download .ipynb** rồi thay
   `colab/Lab22_DPO_T4.ipynb` ở máy local; đây là bản chứa output cells.
2. Copy `data/eval/`, `submission/screenshots/`, Reflection và các file nhỏ từ
   `MyDrive/lab22-artifacts/` về đúng đường dẫn trong repo.
3. Force-add artifact nhỏ bị ignore; không force-add model weights:

   ```bash
   git add -A
   git add -f adapters/sft-mini/adapter_config.json
   git add -f adapters/dpo/adapter_config.json adapters/dpo/dpo_metrics.json
   git add -f adapters/dpo-b0.05/dpo_metrics.json adapters/dpo-b0.50/dpo_metrics.json
   git add -f data/pref/train.parquet
   ```

4. Không commit `adapter_model.safetensors`, `merged-fp16/`, `.env`, key hoặc
   GGUF. Giữ GGUF trong Drive để đối chứng.
5. Chạy `python scripts/verify.py --full`, xem `git diff --cached`, rồi mới push
   repo public và nộp URL LMS.
