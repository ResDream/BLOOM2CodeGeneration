import json
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from tqdm import tqdm

# 加载模型和分词器
model_path = "bigscience/bloom-3b"
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 将模型移动到GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 读取数据集文件
input_file_path = "./data/human-eval.jsonl"
output_file_path = "./results/human-eval-output-transformers.jsonl"

# 定义生成样本的数量
k = 100

# 打开输出文件
with open(output_file_path, "w") as output_file:
    # 逐行读取输入文件
    with open(input_file_path, "r") as input_file:
        lines = input_file.readlines()
        for line in tqdm(lines, desc="Generating Code"):
            # 解析每一行的JSON数据
            data = json.loads(line)
            task_id = data["task_id"]
            prompt = data["prompt"]

            # 生成多个代码样本
            for _ in range(k):
                inputs = tokenizer(prompt, return_tensors="pt")
                # 将输入张量移动到GPU
                inputs = {key: value.to(device) for key, value in inputs.items()}

                outputs = model.generate(
                    inputs["input_ids"].to(device),
                    do_sample=True, max_length=2048,
                    temperature=0.8, top_k=3, repetition_penalty=1.5
                )
                completion = tokenizer.decode(outputs[0], skip_special_tokens=True)
                try:
                    print("输出为：" + completion[len(prompt):])
                except:
                    print("输出为空！！！")

                # 构建输出结果
                output_data = {
                    "task_id": task_id,
                    "completion": completion
                }

                # 将结果写入输出文件
                output_file.write(json.dumps(output_data) + "\n")

print("生成完成，结果已保存到", output_file_path)