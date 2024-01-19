from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("VietAI/vit5-base")  
model = AutoModelForSeq2SeqLM.from_pretrained("VietAI/vit5-base")

num_added_toks = tokenizer.add_tokens([f'<{i}>' for i in range(1921)] + ["<SEG>", "</SEG>", "</POS>"], special_tokens=True)
model.resize_token_embeddings(len(tokenizer))
print(tokenizer)

# sentence = '<SEG><p>1,2,3,4,5,6</p>Khoảng 1.400 công nhân, kỹ sư cùng gần 400 phương tiện, thiết bị được huy động thi công ga T3 nhằm đẩy nhanh tiến độ, hoàn thành dự án vào năm sau.</SEG>'
# encoding = tokenizer(sentence, return_tensors="pt")
# input_ids, attention_masks = encoding["input_ids"].to("cpu"), encoding["attention_mask"].to("cpu")
# print(input_ids)

sentence = """
Khoảng 1.400 công nhân, kỹ sư cùng gần 400 phương tiện, thiết bị được huy động thi công ga T3 nhằm đẩy nhanh tiến độ, hoàn thành dự án vào năm sau.
Thông tin được ông Lê Khắc Hồng, Trưởng Ban quản lý dự án xây dựng ga T3 sân bay Tân Sơn Nhất, cho biết sáng 19/1, khi đề cập tiến độ công trình sau hơn một năm khởi công.
Đây là ga phục vụ khách nội địa quy mô lớn nhất nước, công suất 20 triệu khách mỗi năm, được đầu tư gần 11.000 tỷ đồng từ nguồn vốn của Tổng công ty hàng không (ACV).
Theo ông Hồng, sau khi khởi công cuối năm 2022, hiện công trình đã hoàn thành toàn bộ phần phá dỡ, nền đất, móng cọc, sàn đáy tầng hầm.
Trên công trường, nhà thầu đang đồng loạt triển khai các hạng mục chính là ga hành khách, nhà xe, trung tâm dịch vụ phi hàng không... 
Những hạng mục này hiện đạt khoảng 50% phần thô, trong đó ga hành khách dự kiến hoàn thiện kết cấu vào tháng 5 năm nay, nhà để xe sẽ xong sau đó ba tháng.
"Để đảm bảo kế hoạch, 5 nhà thầu của dự án hiện huy động khoảng 1.400 công nhân, kỹ sư, 16 cẩu tháp cùng 350 đầu xe, phương tiện trên công trường", ông Hồng nói và cho biết quá trình thi công được tổ chức, kiểm soát chặt về tiến độ theo chu kỳ 15 ngày. 
Những phần việc chậm trễ, nhà thầu phải có giải pháp bù lại tiến độ để bám sát kế hoạch chung.
Đại diện chủ đầu tư cũng cho biết do mặt bằng chật hẹp khiến việc tổ chức thi công, lắp dựng cẩu, đường công vụ, bãi vật liệu, lán trại công nhân... gặp khó khăn. 
Các nhà thầu phải luân phiên, linh hoạt điều chỉnh vị trí tuỳ theo tiến độ công trình.
Ngoài ra, dự án nhà ga cũng đang triển khai đồng thời công trình đường nối Trần Quốc Hoàn - Cộng Hoà bên ngoài, nên giữa hai chủ đầu tư phải thường xuyên phối hợp, lên kế hoạch đồng bộ trong quá trình vận chuyển vật liệu, chất thải, thiết bị.
"Tuy nhiên, tiến độ thi công đến thời điểm này đang kiểm soát tốt, đảm bảo theo kế hoạch hoàn thành toàn bộ dự án giữa năm 2025", ông Hồng nói.
"""

sentence = '<SEG><1><2><3><4><5><6></POS>' + sentence + '</SEG>'
encoding = tokenizer(sentence, return_tensors="pt")
input_ids, attention_masks = encoding["input_ids"].to("cpu"), encoding["attention_mask"].to("cpu")

print(input_ids)
print(len(sentence))
print(input_ids.shape)

# outputs = model.generate(
#     input_ids=input_ids, attention_mask=attention_masks,
#     max_length=1024,
#     early_stopping=True
# )

# for output in outputs:
#     line = tokenizer.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True)
#     print(line)
#     print(len(line))