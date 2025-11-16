import pandas as pd


def evaluation_true_false(answers_path, results_path):
    answers = pd.read_json(answers_path, lines=True, orient="records")
    results = pd.read_json(results_path)
    assert len(answers) == len(results)
    sum = 0
    total = len(answers)
    for index, row in answers.iterrows():
        assert row["Question"].lower() == results.iloc[index]["question"].split("Only give a boolean answer, False or True.")[0].rstrip().lower()
        if row["Answer"].lower() == results.iloc[index]["answer"].lower():
            sum += 1
    return sum/total


if __name__ == "__main__":
    print(evaluation_true_false("./dataset/questions/TF.jsonl","./results/qa_20_filter_TF.json"))
