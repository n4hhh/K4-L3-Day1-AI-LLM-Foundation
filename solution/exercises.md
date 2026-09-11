# K4 — Ngày 1: Bài Tập & Phản Ánh
## Khám Phá LLM API | Phiếu Thực Hành

**Thời lượng:** 4 tiếng
**Cách làm:** Trả lời từng câu ngay sau khi hoàn thành block tương ứng —
đừng để dồn hết về cuối buổi. Thay dòng `*Câu trả lời của bạn*` bằng câu
trả lời thật (chấm tự động sẽ đếm số câu đã trả lời).

---

## Block 1 — API Cơ Bản (trả lời sau Checkpoint 1)

### Câu 1.1 — Độ nhạy của temperature
Gọi `call_openai` với temperature 0.0, 0.5, 1.0 và 1.5 dùng prompt
**"Hãy kể cho tôi một sự thật thú vị về Việt Nam."**

**Bạn nhận thấy quy luật gì qua bốn phản hồi?** (2–3 câu)
Khi temperature tăng từ 0.0 lên 1.5, phản hồi có xu hướng đa dạng và tự do hơn về cách diễn đạt cũng như nội dung; ở mức cao, model mở rộng câu trả lời và đưa ra nhiều chi tiết hơn thay vì chỉ tập trung vào một ý. Tuy nhiên, các phản hồi cũng xuất hiện nhiều thông tin thiếu chính xác hoặc có dấu hiệu “hallucination”, cho thấy temperature cao có thể làm tăng tính sáng tạo nhưng giảm độ ổn định và đáng tin cậy của câu trả lời. Temperature thấp giúp đầu ra ổn định hơn, nhưng không đảm bảo thông tin luôn chính xác.

### Câu 1.2 — Chọn temperature cho sản phẩm
**Bạn sẽ đặt temperature bao nhiêu cho chatbot hỗ trợ khách hàng, và tại sao?**
Mình sẽ chọn temperature khoảng 0.2–0.3 cho chatbot hỗ trợ khách hàng. Chatbot cần ưu tiên tính chính xác, nhất quán và hạn chế hallucination hơn là sự sáng tạo, vì vậy temperature thấp sẽ phù hợp hơn.

### Câu 1.3 — Đánh đổi chi phí
Kịch bản: 10.000 người dùng hoạt động mỗi ngày, mỗi người gọi API 3 lần,
mỗi lần trung bình ~350 token đầu ra.

**Ước tính GPT-4o đắt hơn GPT-4o-mini bao nhiêu lần cho workload này? Nêu một
trường hợp GPT-4o xứng đáng với chi phí và một trường hợp nên dùng mini:**

GPT-4o có giá output $0.010/1K tokens, còn GPT-4o-mini là $0.0006/1K tokens. Vì vậy, xét workload đầu ra này, GPT-4o đắt hơn khoảng 16,7 lần, tương ứng khoảng $105/ngày so với $6,3/ngày.
GPT-4o xứng đáng với chi phí cho các tác vụ phức tạp cần chất lượng và khả năng suy luận cao, chẳng hạn phân tích một yêu cầu kỹ thuật phức tạp của khách hàng. GPT-4o-mini phù hợp hơn cho các tác vụ đơn giản với số lượng lớn như trả lời FAQ, phân loại yêu cầu hoặc xử lý các câu hỏi hỗ trợ thông thường.
---

## Block 2 — System Prompt & Token (trả lời sau Checkpoint 2)

### Câu 2.1 — Sức mạnh của persona
Gọi `chat_with_system_prompt` hai lần với cùng câu hỏi
**"Giải thích blockchain là gì?"** nhưng hai system prompt khác nhau:
- "Bạn là giáo viên tiểu học, giải thích thật đơn giản cho trẻ 8 tuổi."
- "Bạn là chuyên gia tài chính, trả lời chuyên sâu bằng thuật ngữ kỹ thuật."

**Hai phản hồi khác nhau như thế nào (độ dài, từ vựng, ví dụ)? System prompt
ảnh hưởng đến hành vi model ra sao?** (3–4 câu)

Hai phản hồi khác nhau rõ rệt về độ dài, từ vựng và mức độ kỹ thuật. Với persona giáo viên tiểu học, model trả lời ngắn gọn, sử dụng từ đơn giản và ví blockchain như một “cuốn sổ lớn” để trẻ dễ hình dung; trong khi persona chuyên gia tài chính tạo câu trả lời dài và chuyên sâu hơn với các thuật ngữ như hash, encryption và consensus. Điều này cho thấy system prompt có thể định hướng rõ rệt cách model trình bày cùng một kiến thức, bao gồm mức độ chi tiết, lựa chọn từ vựng và loại ví dụ phù hợp với đối tượng người đọc.


### Câu 2.2 — tiktoken vs đếm từ
Chọn một đoạn văn tiếng Việt ~100 từ. So sánh số token theo `count_tokens`
(tiktoken) với ước lượng `số từ / 0.75` mà Part 1 đã dùng.

**Hai con số chênh nhau bao nhiêu phần trăm? Vì sao tiếng Việt thường tốn
nhiều token hơn tiếng Anh cùng độ dài?**

Đoạn văn tiếng Việt gồm 109 từ. Theo công thức ước lượng số từ / 0.75, kết quả là khoảng 145,33 token, trong khi tiktoken đếm được 126 token, chênh lệch khoảng 13.3% ; trong trường hợp này phương pháp đếm từ đã ước lượng cao hơn thực tế.
Tiếng Việt thường tốn nhiều token hơn tiếng Anh có nội dung tương đương vì tokenizer chia văn bản thành các đơn vị subword thay vì đếm từ trực tiếp. Tiếng Việt có nhiều dấu thanh, ký tự có dấu và các từ ghép gồm nhiều âm tiết được viết cách nhau, khiến một từ về mặt ngữ nghĩa có thể được biểu diễn bởi nhiều token. Vì vậy, sử dụng tokenizer thực tế sẽ cho kết quả chính xác hơn so với quy tắc ước lượng dựa trên số từ.


---

## Block 3 — Streaming & Độ Bền (trả lời sau Checkpoint 3)

### Câu 3.1 — Trải nghiệm người dùng với streaming
**Streaming quan trọng nhất trong trường hợp nào, và khi nào thì
non-streaming lại phù hợp hơn?** (1 đoạn văn)

Streaming quan trọng nhất với các ứng dụng tương tác như chatbot hoặc AI assistant, đặc biệt khi model tạo câu trả lời dài, vì người dùng có thể thấy nội dung ngay khi model bắt đầu sinh thay vì phải chờ toàn bộ phản hồi hoàn thành. Điều này làm giảm thời gian chờ cảm nhận (perceived latency) và tạo cảm giác hệ thống phản hồi nhanh hơn. Non-streaming phù hợp hơn khi cần nhận toàn bộ kết quả trước khi xử lý, chẳng hạn phân loại văn bản, trích xuất dữ liệu có cấu trúc, xử lý batch hoặc khi response rất ngắn.

### Câu 3.2 — Vì sao backoff theo cấp số nhân?
**So với delay cố định (ví dụ luôn chờ 1 giây), exponential backoff có lợi
thế gì khi API bị quá tải? Điều gì xảy ra nếu hàng nghìn client cùng retry
với delay cố định giống nhau?**

Exponential backoff tăng dần thời gian chờ sau mỗi lần request thất bại, ví dụ 1s → 2s → 4s → 8s, giúp giảm số request gửi đến server khi API đang quá tải và cho hệ thống thời gian phục hồi. Nếu hàng nghìn client đều sử dụng delay cố định 1 giây, chúng có thể retry gần như cùng lúc sau mỗi giây, tạo ra các đợt request lớn lặp lại và khiến server tiếp tục quá tải, hiện tượng này thường được gọi là thundering herd. Exponential backoff, đặc biệt khi kết hợp thêm random jitter, giúp phân tán thời điểm retry giữa các client và giảm nguy cơ quá tải lặp lại.

---

## Block 4 — Mini-Project (trả lời sau Checkpoint 4)

### Câu 4.1 — Thiết kế persona
**Bạn chọn persona gì cho trợ lý của mình? Viết lại system prompt đó và giải
thích 1–2 lựa chọn từ ngữ quan trọng trong prompt (ví dụ: vì sao yêu cầu
"trả lời ngắn gọn", vì sao chỉ định ngôn ngữ...):**

“Bạn là trợ lý học tập chuyên về lập trình và trí tuệ nhân tạo. Hãy trả lời bằng tiếng Việt, ngắn gọn và dễ hiểu. Khi giải thích khái niệm kỹ thuật, hãy đưa ra ví dụ thực tế hoặc đoạn code ngắn khi phù hợp. Nếu không chắc chắn về thông tin, hãy nói rõ thay vì tự tạo ra câu trả lời.”
Mình yêu cầu “ngắn gọn và dễ hiểu” để tránh việc model tạo ra những câu trả lời quá dài, đặc biệt khi người dùng chỉ cần giải thích nhanh một khái niệm. Yêu cầu “nếu không chắc chắn hãy nói rõ” nhằm hạn chế hallucination, vì độ chính xác quan trọng đối với một trợ lý dùng để học các kiến thức kỹ thuật.

### Câu 4.2 — Hạn chế & cải thiện
**Trợ lý của bạn hiện có hạn chế lớn nhất là gì (ví dụ: history chỉ 3 lượt,
không có bộ nhớ dài hạn, không kiểm duyệt nội dung...)? Đề xuất một cải
thiện cụ thể và mô tả ngắn cách triển khai:**

Hạn chế lớn nhất của trợ lý hiện tại là chỉ lưu 3 lượt hội thoại gần nhất, vì vậy model có thể quên những thông tin hoặc yêu cầu quan trọng đã xuất hiện trước đó khi cuộc hội thoại kéo dài. Một cải thiện cụ thể là bổ sung bộ nhớ dài hạn bằng retrieval (RAG): lưu các nội dung quan trọng của lịch sử hội thoại dưới dạng embedding trong vector database, sau đó tìm kiếm những thông tin liên quan đến câu hỏi hiện tại và đưa chúng trở lại context trước khi gọi model. Cách này giúp trợ lý duy trì được thông tin cần thiết mà không phải gửi toàn bộ lịch sử hội thoại trong mỗi request.

---

## Danh Sách Kiểm Tra Nộp Bài

- [ ] `python grade.py` — xem điểm tự động, mục tiêu ≥ 75/100
- [ ] Cả 4 checkpoint pytest đều pass
- [ ] Tất cả 9 câu trong file này đã được trả lời
- [ ] Đã copy bài làm vào folder `solution/`, push lên fork và dán link trên trang bài Lab ở VLearn trước 23:59 ngày 11/09/2026
