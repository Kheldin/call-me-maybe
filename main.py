from llm_sdk import Small_LLM_Model

def main():
    print("----------------------------------------------------------------------------------")
    llm = Small_LLM_Model()
    tokenized_str = llm.encode("Je suis une string")
    print(tokenized_str)
    print(llm.decode(tokenized_str))



if __name__ == "__main__":
    main()
