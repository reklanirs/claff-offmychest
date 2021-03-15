from header import *
from keras_text.data import Dataset
from keras_text.processing import WordTokenizer



def main():
    X, Y = read_data(training_data_path, clean=False)
    final_test_sentenceid, final_test_text = extract_columns(pd.read_csv(test_data_path, header=0), ['sentenceid', 'full_text'])

    tokenizer = WordTokenizer()
    pass


if __name__ == '__main__':
    main()