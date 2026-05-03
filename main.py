from llm_sdk import Small_LLM_Model
import numpy as np
import json


def main():
    print("----------------------------------------------------------------------------------")
    sentence = "Salut c'est Antoine "
    llm = Small_LLM_Model()
    path = llm.get_path_to_vocab_file()
    tokenized_str = llm.encode(sentence)
    list_logits = llm.get_logits_from_input_ids(list(tokenized_str[0]))
    # print()
    # print(tokenized_str)
    # print(llm.decode(tokenized_str))
    maxx = np.argmax(list_logits)
    try: 
        with open(path, "r") as file:
            dict_json = json.load(file)
            for i in range(50):
                list_logits = llm.get_logits_from_input_ids(list(tokenized_str[0]))
                for key, value in dict_json.items():
                    if (value == maxx):
                        key_bis = key
                        key_bis = key_bis.replace("Ġ", " ")
                sentence = sentence + key_bis
                tokenized_str = llm.encode(sentence)
                maxx = np.argmax(list_logits)
        print(sentence)
    except Exception as e:
        print(e)
    # check BPE


if __name__ == "__main__":
    main()
