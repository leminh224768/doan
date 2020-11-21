from flask import Flask, render_template,request,send_file,Response
import pickle
import numpy as np
import os, string, re
from sklearn.feature_extraction.text import TfidfVectorizer
from pyvi import ViTokenizer, ViPosTagger
import os
app = Flask(__name__)


# Load hàm tiền xử lý dữ liệu. Cái hàm này được lấy từ trên quá trình huấn luyện ở file code google colab
classifier_path = "svm_classifier.pkl"  
with open(classifier_path, 'rb') as f:
    classifier = pickle.load(f)
    
path = "x_train_featured.pkl"  
with open(path, 'rb') as f:
    x_train_featured = pickle.load(f)
    
vectorizer = TfidfVectorizer(preprocessor=lambda x: x, tokenizer=lambda x: x)
vectorizer.fit(x_train_featured)    


def clean_text(text):
    # Xóa bỏ dấu câu
    daucau = "!\"#$%&'()*,./:;<=>?^_`~"
    for punc in daucau.split():
        text = text.replace(punc,' '+ punc + ' ')
    # Tách từ
    text = ViTokenizer.tokenize(text)
    # Dua về dạng chữ thường
    text = text.lower()
    text = re.sub('\\s+',' ',text)
    return text


# Cái đoạn code sau đây các phương pháp rút trích đặc trưng
# Đây là đoạn code rút trích ra các feature: ngram, từ loại : danh từ , động từ tính từ.
def ngram_featue(text,N):
    sentence = text.split(" ")
    grams = [sentence[i:i+N] for i in range(len(sentence)-N+1)]
    result = []
    for gram in grams:
        result.append(" ".join(gram))
    return result

# Rút trích từ vựng dựa trên nhãn từ loại
def get_Word_based_POS(text):
    tag_pos = ViPosTagger.postagging(text)
    vocab = tag_pos[0]
    list_pos = tag_pos[1]
    result = []
    for index,pos in enumerate(list_pos):
        if "N" in pos or "V" in pos or "A" in pos:
            result.append(vocab[index])
    return result

# Lấy đặc trưng nhãn từ loại.
def get_POS_feature(text):
    tag_pos = ViPosTagger.postagging(text)
    vocab = tag_pos[0]
    list_pos = tag_pos[1]
    result = []
    for index,pos in enumerate(list_pos):
        result.append(pos)
    return result

# Hàm rút trích đặc trưng chính.
def extract_feature(text_preproced):
    feature = ngram_featue(text_preproced,2) + ngram_featue(text_preproced,3) + ngram_featue(text_preproced,4)
    feature += get_Word_based_POS(text_preproced) + get_POS_feature(text_preproced)
    return feature


int2labels = {0: "Tiêu cực", 1: "Trung Tính", 2: "Tích cực"}

    

# Đây là đoạn code xử lý code xây dựng ứng dụng: nếu địa chỉ IP vào là 127.0.0:5000 là sẽ trả file html "Home"
@app.route("/")
def home():
    return render_template("home.html")

@app.route("/dulieu")
def dulieu():
    return render_template("dulieu.html")


@app.route("/ketqua")
def phantichketqua():
    return render_template("ketqua.html")

@app.route("/phantich/", methods=['POST','GET'])
def phantich():
    query_value = request.form['query']
    input_process = clean_text(query_value)
    input_feature = [extract_feature(input_process)]
    input_tfidf = vectorizer.transform(input_feature)
    predicted = classifier.predict(input_tfidf)
    label = int2labels[predicted[0]]
    print("Kết quả dự đoán : ", label)
    return render_template("result.html", data = [{"label":label,"query":query_value}])


# Hàm main chạy chương trình web Flask trên local
if __name__ == "__main__":
    app.run()
