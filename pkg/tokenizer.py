from deepchem.feat.smiles_tokenizer import SmilesTokenizer
import pandas as pd



if __name__ == "__main__":
    tokenizer = SmilesTokenizer(vocab_file="data/vocab.txt")
    
    test_data = pd.read_csv("data/train_meta.csv")
    
    test_data = test_data["smiles"].max()


    # print(tokenizer.vocab_size)
    # print(tokenizer.add_padding_tokens(tokenizer.encode(test_data), length=500))

    # print(tokenizer.decode(tokenizer.encode(test_data["smiles"].iloc[0])))