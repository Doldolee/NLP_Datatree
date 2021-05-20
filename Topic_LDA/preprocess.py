from eunjeon import Mecab
from tqdm import tqdm 
import re,pickle,csv
import pandas as pd


###전처리

#온점, 반점 제거, cleantext
def clean_text(text):
    text = text.replace(".","").strip()
    text = text.replace(",","").strip()
    pattern = '[^ ㄱ-ㅣ가-힣|0-9|a-zA-Z]+'
    text = re.sub(pattern=pattern, repl='', string=text)
    return text

#명사 추출
def get_nouns(tokenizer, sentence):
    tagged = tokenizer.pos(sentence)
    nouns = [s for s, t in tagged if t in ['SL','NNG','NNP'] and len(s)>1]
    return nouns

#토큰화
def tokenize(df):
    tokenizer = Mecab()
    processed_data = []
    for sent in tqdm(df['description']):
        sentence = clean_text(sent.replace("\n","").strip())
        processed_data.append(get_nouns(tokenizer,sentence))
    return processed_data

#저장
def save_processed_data(processed_data):
    with open("./tokenized_data.csv",'w',newline='',encoding='utf-8') as f:
        writer = csv.writer(f)
        for data in processed_data:
            writer.writerow(data)

#실행
if __name__ == '__main__':
    df = pd.read_csv("historical_records.csv")
    processed_data = tokenize(df)

    save_processed_data(processed_data)
