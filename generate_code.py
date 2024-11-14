# from mindnlp.transformers import AutoModelForCausalLM, AutoTokenizer
# import mindspore
#
# model_path = "bigscience/bloom-1b7"
#
# model = AutoModelForCausalLM.from_pretrained(model_path, ms_dtype=mindspore.float16)
# tokenizer = AutoTokenizer.from_pretrained(model_path)
#
# # content = """
# # from typing import List\n\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    \"\"\" Check if in given list of numbers, are any two numbers closer to each other than\n    given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    \"\"\"\n", "entry_point": "has_close_elements", "canonical_solution": "    for idx, elem in enumerate(numbers):\n        for idx2, elem2 in enumerate(numbers):\n            if idx != idx2:\n                distance = abs(elem - elem2)\n                if distance < threshold:\n                    return True\n\n    return False\n", "test": "\n\nMETADATA = {\n    'author': 'jt',\n    'dataset': 'test'\n}\n\n\ndef check(candidate):\n    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3) == True\n    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.05) == False\n    assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.95) == True\n    assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.8) == False\n    assert candidate([1.0, 2.0, 3.0, 4.0, 5.0, 2.0], 0.1) == True\n    assert candidate([1.1, 2.2, 3.1, 4.1, 5.1], 1.0) == True\n    assert candidate([1.1, 2.2, 3.1, 4.1, 5.1], 0.5) == False\n\n
# # """
#
# content = f'This is a creative story about An Unexpected Journey Through Time.\n'
# inputs = tokenizer(content, return_tensors="ms")
# # outputs = model.generate(
# #     inputs.input_ids,
# #     max_length=1024,
# #     do_sample=False,
# #     repetition_penalty=2.0
# # )
# outputs = model.generate(
#     inputs.input_ids,
#     do_sample=True, max_length=200, top_k=1,
#     temperature=0, repetition_penalty=2.0
# )
# response = tokenizer.decode(outputs[0], skip_special_tokens=True)
# print(response)

import json
from mindnlp.transformers import AutoModelForCausalLM, AutoTokenizer
import mindspore
from tqdm import tqdm

# 加载模型和分词器
model_path = "bigscience/bloom-1b7"
model = AutoModelForCausalLM.from_pretrained(model_path, ms_dtype=mindspore.float16)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 读取数据集文件
input_file_path = "./data/human-eval.jsonl"
output_file_path = "./results/human-eval-output.jsonl"

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

            # 使用模型生成代码
            inputs = tokenizer(prompt, return_tensors="ms")
            outputs = model.generate(
                inputs.input_ids,
                do_sample=True, max_length=1024, top_k=1,
                temperature=0, repetition_penalty=1.5
            )
            completion = tokenizer.decode(outputs[0], skip_special_tokens=True)

            # 构建输出结果
            output_data = {
                "task_id": task_id,
                "completion": completion
            }

            # 将结果写入输出文件
            output_file.write(json.dumps(output_data) + "\n")

print("生成完成，结果已保存到", output_file_path)