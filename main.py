from llm_sdk import Small_LLM_Model
import numpy as np
import json


def main():
    print("----------------------------------------------------------------------------------")
    llm = Small_LLM_Model()
    path = llm.get_path_to_vocab_file()
    tokenized_str = llm.encode("Je suis une string")
    list_logits = llm.get_logits_from_input_ids(list(tokenized_str[0]))
    maxx = np.argmax(list_logits)
    # print()
    # print(tokenized_str)
    # print(llm.decode(tokenized_str))
    try: 
        with open(path, "r") as file:
            dict_json = json.load(file)
            print(dict_json)
    except Exception as e:
        print(e)
    # check BPE


if __name__ == "__main__":
    main()
