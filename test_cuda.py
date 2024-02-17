from datasets import load_dataset

dataset = load_dataset("wmt16", "cs-en")

# Truy cập vào các phần dữ liệu trong bộ dữ liệu
train_data = dataset["train"]
validation_data = dataset["validation"]
test_data = dataset["test"]

for item in test_data:
    print(item['translation']['cs'])
