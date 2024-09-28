from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer
import torch
from transformers import logging

logging.set_verbosity_error()

# 设置模型路径
model_path = "/home/dongsheng/Model/dpr-question_encoder-single-nq-base/"

# 加载 tokenizer 和模型
tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(model_path)
model = DPRQuestionEncoder.from_pretrained(model_path)

# 设置设备为 cuda:1（如果可用）
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print("running on", device)

# 将模型移动到指定设备
model = model.to(device)
model.eval()

# 对输入文本进行 tokenization 并将输入张量移动到相同设备
sentences = ["Hello, is my dog cute?", "How are you doing?"]
input_ids = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt", max_length=512)["input_ids"].to(device)

# 计算 embeddings
with torch.no_grad():  # 确保模型在推理模式下，不计算梯度
    embeddings = model(input_ids).pooler_output

# 打印结果
# print(embeddings)
print(embeddings.shape)
