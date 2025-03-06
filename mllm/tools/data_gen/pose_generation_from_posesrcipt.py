import json
import random
import argparse
import os
import string

questions = [
    "I have a word description of a person's pose: {description}. Can you give the SMPL pose of this person?",
    "{description}. Give the SMPL pose.",
    "{description}. What's the SMPL pose of this person? ",
    "{description}. Use SMPL pose to describe this person's behavior.",
    "There is a person: {description}. Please output this person's SMPL pose.",
    "There is a person doing this: {description}. Can you use SMPL pose to describe the pose?",
    "A person is described as: {description}. Use the SMPL pose to reflect this.",
    "Human pose is described as words: {description}. The SMPL pose is?",
    "Human pose can be described as words: {description}. And it can also be described in SMPL pose format, can you output this?"
]

answers = [
    "The SMPL pose is <POSE>.",
    "It is <POSE>.",
    "Sure, it is <POSE>.",
    "Sure, the SMPL pose is <POSE>.",
    "The SMPL pose of the person is <POSE>.",
    "Sure, <POSE>."
]


def capitalize_first_letter(input_string):
    if not input_string:
        return input_string

    # 将第一个字母大写转换为小写,
    result_string = input_string[0].lower() + remove_punctuation_at_end(input_string[1:])
    return result_string


def remove_punctuation_at_end(input_str):
    # 获取标点符号集合
    punctuation_set = set(string.punctuation)

    # 检查字符串是否以标点符号结尾
    if input_str and input_str[-1] in punctuation_set:
        # 如果是，去掉结尾的标点符号
        input_str = input_str[:-1]

    return input_str


def create_directory_if_not_exists(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)


def split_train_test(data_json_file, save_json_dir, num_train_samples=0.8, prefix=None):
    with open(data_json_file, 'r') as f:
        all_data = json.load(f)
        print(f"数据集共有{len(all_data)}条数据")

    if isinstance(num_train_samples, float):
        assert 0 < num_train_samples < 1, "num_train_samples should be between 0 and 1"
        num_train_samples = int(len(all_data) * num_train_samples)
    else:
        assert isinstance(num_train_samples, int), "num_train_samples must be int type"

    train_data = all_data[:num_train_samples]
    train_json_path = os.path.join(save_json_dir, f"{prefix}_train.json" if prefix is not None else "train.json")
    generate_text2pose(train_data, train_json_path)

    if num_train_samples < len(all_data):
        test_data = all_data[num_train_samples:]
        test_json_path = os.path.join(save_json_dir, f"{prefix}_test.json" if prefix is not None else "test.json")
        generate_text2pose(test_data, test_json_path)
    else:
        print("num_train_samples greater than the length of dataset, no test dataset")


def generate_text2pose(all_data, save_json_file):
    result = []
    for data in all_data:
        text_description = data['text']
        question = random.choice(questions)
        if question.startswith("{description}"):
            question = question.format(description=remove_punctuation_at_end(text_description))
        else:
            question = question.format(description=capitalize_first_letter(text_description))
        answer = random.choice(answers)
        conversation = [{"from": "human", "value": question},
                        {"from": "gpt", "value": answer}]
        single_data = {
            "id": data['id'],
            "pose": data['pose'],
            "conversations": conversation
        }
        result.append(single_data)

    with open(save_json_file, 'w') as f:
        json.dump(result, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-data-path', type=str,
                        help='posescript预处理文件存放路径',
                        default='data_handler/posecript_pose_rec.json')
    parser.add_argument('--save-json-path', type=str,
                        help='Text2pose json文件存放路径',
                        default='data_handler/text2pose/')
    args = parser.parse_args()
    input_file = args.input_data_path
    output_file = args.save_json_path
    create_directory_if_not_exists(output_file)
    split_train_test(input_file, output_file, num_train_samples=6000, prefix='text2pose')