from flask import Flask, request, render_template, redirect, url_for, jsonify 
import os
# import cv2
import numpy as np
import base64
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
import pandas as pd
import requests
from flask_cors import CORS 
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import numpy as np
import tensorflow as tf
import mysql.connector
import joblib
import re
from datetime import datetime
import torch.nn.functional as F
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import csv
import pickle
from azure.storage.blob import BlobServiceClient

app = Flask(__name__)
CORS(app)
print(tf.__version__)
# CORS(app, resources={r"/predict": {"origins": "http://localhost:3000"}})


# Mô hình nhận diện hình ảnh với RestNet ---------------------------------------------------------------------------------------------------------------------------
# resnet_model = Sequential()
# pretrained_model = tf.keras.applications.ResNet50(
#     include_top=False,
#     input_shape=(180, 180, 3),
#     pooling='avg',
#     weights='imagenet'
# )
# for layer in pretrained_model.layers:
#     layer.trainable = False
# resnet_model.add(pretrained_model)
# resnet_model.add(Flatten())
# resnet_model.add(Dense(512, activation='relu'))
# resnet_model.add(Dense(5, activation='softmax'))
# resnet_model.load_weights('ImageDetect/resnet_model_weights.weights.h5')
# resnet_model.summary()

# img_height, img_width = 180, 180
# class_names = ['Gà rán', 'Cocacola', 'Hamburger', 'Kem', 'Trà sữa']

# @app.route('/predict', methods=['POST'])
# def predict_image():
#     data = request.json
#     if 'image' not in data:
#         return jsonify({'error': 'No image provided'}), 400

#     image_data = data['image']
#     img_data = base64.b64decode(image_data)
#     np_arr = np.frombuffer(img_data, np.uint8)
#     image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)    
#     if image is None:
#         return jsonify({'error': 'Could not decode image'}), 400
    
#     image_resized = cv2.resize(image, (img_height, img_width))
#     image_array = np.expand_dims(image_resized, axis=0)

#     pred = resnet_model.predict(image_array)
#     output_class = class_names[np.argmax(pred)]
#     return output_class

# Mô hình Cross sell predict -----------------------------------------------------------------------------------------------------------------------------------
def connect_db():
    global db_connection
    if db_connection is None or not db_connection.is_connected():
        db_connection = mysql.connector.connect(
            host="dbpbl.mysql.database.azure.com",
            port=3306,
            user="adminn",
            password="Root123456789",
            database="db_pbl6",
            ssl_disabled=False
        )
    return db_connection

db_connection = None

def predict_ratings_vectorized(user, user_product_matrix, similarity_matrix):
    user_ratings = user_product_matrix.loc[user]
    unrated_items = user_ratings[user_ratings == 0].index
    
    user_avg_rating = user_ratings[user_ratings > 0].mean()
    print("user_avg_rating: ",user_avg_rating)
    similarities = similarity_matrix[user]
    
    predictions = []
    for item in unrated_items:
        item_ratings = user_product_matrix[item]
        rated_users = item_ratings[item_ratings > 0].index
        
        if len(rated_users) > 0:
            numerator = np.sum(
                similarities[rated_users] * (item_ratings[rated_users] - user_product_matrix.loc[rated_users].mean(axis=1))
            )
            denominator = np.sum(np.abs(similarities[rated_users]))

            predicted_rating = user_avg_rating + numerator / (denominator + 1e-10)
            predictions.append((item, np.clip(predicted_rating, 1, 5)))
    listproductId = []
    predicted_results_sorted = sorted(predictions, key=lambda x: x[1], reverse=True)[:10]
    # Thêm 10 productId đầu tiên từ predicted_results_sorted vào listproductId
    for item in predicted_results_sorted[:10]:
        listproductId.append(item[0])   
    return listproductId



@app.route('/cross-sell/<int:userId>', methods=['GET'])
def crossSell(userId):
    user_predict = userId
    print(userId)
    with open('myflask/user_product_matrix.pkl', 'rb') as f:    
        user_product_matrix = pickle.load(f)
    with open('myflask/similarity_matrix.pkl', 'rb') as f:
        similarity_matrix = pickle.load(f)
    try:
        predictions = predict_ratings_vectorized(userId, user_product_matrix, similarity_matrix)
        print(predictions)
        return jsonify(predictions)
    except ValueError as e:
        print(e)
        return None



def selectProduct():
    db_connection = connect_db()
    cursor = db_connection.cursor(dictionary=True)
    query = "select product_name from product"
    cursor.execute(query)
    r = cursor.fetchall()
    result = [row['product_name'] for row in r]
    return result

def extract_product_name(question):
    result = selectProduct()
    question_lower = question.lower()  # Chuyển câu hỏi thành chữ thường
    
    for product_name in result:
        if product_name.lower() in question_lower:
            return product_name
    return None   

def fetch_data(intent, question,storeId):
    db_connection = connect_db()    
    cursor = db_connection.cursor(dictionary=True)
    response = "Xin lỗi, tôi không tìm thấy thông tin phù hợp."
    
    # Xử lý ý định "product_price"
    if intent == "product_price":
        product_name = extract_product_name(question)
        print("product_name: ",product_name)        
        if product_name:
            # query = "SELECT price FROM product WHERE product_name = %s"
            # cursor.execute(query2, (product_name,))
            query2 = "SELECT discounted_price FROM product WHERE product_name LIKE %s"
            value = f"%{product_name}%"
            cursor.execute(query2, (value,))
            result = cursor.fetchone()
            if result:
                response = f"Giá của sản phẩm {product_name} là {int(result['discounted_price'])} VND."
            else:
                response = f"Xin lỗi, không tìm thấy sản phẩm {product_name} trong cơ sở dữ liệu."
        else:
            response = "Bạn vui lòng cung cấp tên sản phẩm để tôi có thể tìm giá."

    elif intent == 'product_info':
        product_name = extract_product_name(question)
        print("product_name: ",product_name)
        if product_name:
            query = "SELECT * FROM product WHERE product_name = %s"
            # cursor.execute(query2, (product_name,))
            query2 = "SELECT * FROM product WHERE product_name LIKE %s"
            value = f"%{product_name}%"
            cursor.execute(query2, (value,))
            result = cursor.fetchone()
            if result:
                response = (
                    f"Một số thông tin của sản phẩm {result['product_name']}:\n"
                    f" - Mã sản phẩm: {result['product_id']}\n"
                    f" - Giá: {int(result['price'])} VND\n"
                    f" - Giá được giảm: {int(result['discounted_price'])} VND\n"
                    f" - Mô tả: {result['description']}\n"
                )                       
            else:
                response = f"Xin lỗi, không tìm thấy sản phẩm {product_name} trong cơ sở dữ liệu."
        else:
            response = "Bạn vui lòng cung cấp tên sản phẩm để tôi có thể xem thông tin."

    elif intent == 'opening_hours':
        query = "SELECT opening_time, closing_time FROM store WHERE store_id = %s"
        cursor.execute(query,(storeId,))
        result = cursor.fetchone()
        if result:
            response = f'Cửa hàng của chúng tôi mở cửa vào lúc {result["opening_time"].strftime("%H:%M")} và đóng cửa vào lúc {result["closing_time"].strftime("%H:%M")}'
        else:
            response = f"Xin lỗi, tôi không tìm thấy cửa hàng mà bạn muốn biết thông tin"    
    
    elif intent == 'location':
        query = "SELECT location FROM store WHERE store_id = %s"
        cursor.execute(query,(storeId,))
        result = cursor.fetchone()
        if result:
            response = f"Cửa hàng của chúng tôi có vị trí tại : {result['location']}."
        else:
            response = f"Xin lỗi, tôi không tìm thấy cửa hàng mà bạn muốn biết thông tin"    
       
    elif intent == 'phone_number':
        query = "SELECT phone_number FROM store WHERE store_id = %s"
        cursor.execute(query,(storeId,))
        result = cursor.fetchone()
        if result:
            response = f"Số điện thoại của chủ cửa hàng chúng tôi : {result['phone_number']}."
        else:
            response = f"Xin lỗi, tôi không tìm thấy cửa hàng mà bạn muốn biết thông tin"    
        
    elif intent == "hello":
        response = "Chào bạn, bạn cần sự trợ giúp gì?"
    
    elif intent == "stock_status":
        # Tìm product_id dựa trên tên sản phẩm
        product_name = extract_product_name(question)
        print(product_name)
        query2 = "SELECT product_id FROM product WHERE product_name = %s"
        cursor.execute(query2, (product_name,))
        result = cursor.fetchone()
        
        # Nếu tìm thấy sản phẩm
        if result:
            product_id = result['product_id']
            
            # Kiểm tra số lượng trong kho của sản phẩm ở cửa hàng
            query1 = "SELECT stock_quantity FROM product_store WHERE store_id = %s AND product_id = %s"
            cursor.execute(query1, (storeId, product_id))
            stock_status = cursor.fetchone()

            if stock_status:
                response = f"Số lượng sản phẩm {product_name} còn lại trong kho là: {stock_status['stock_quantity']}"
            else:
                response = "Sản phẩm đã hết trong kho tại cửa hàng này."
        else:
            response = "Không tìm thấy sản phẩm với tên đã cho." 
        
  
    cursor.close()
    return response

# Định nghĩa thiết bị
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load tokenizer và model PhoBERT
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")

# Khởi tạo mô hình giống như lúc huấn luyện
model = AutoModelForSequenceClassification.from_pretrained("vinai/phobert-base", num_labels=7)  # Số nhãn thay đổi tùy dữ liệu của bạn
STORAGE_CONNECTION_STRING = "DefaultEndpointsProtocol=https;AccountName=aimystorage123;AccountKey=m0mEMb+P83wXqit8Nl9MIGcp1xvBCyAALjYzOoJMpHnDEFGzrB3GulYaZMWSAa9Y4snB+jzWz9cI+AStsPY5cw==;EndpointSuffix=core.windows.net"
CONTAINER_NAME = "models"  # Tên container bạn đã tạo

def download_blob(blob_name, download_path):
    """Hàm tải tệp từ Azure Blob Storage."""
    blob_service_client = BlobServiceClient.from_connection_string(STORAGE_CONNECTION_STRING)
    blob_client = blob_service_client.get_blob_client(container=CONTAINER_NAME, blob=blob_name)

    with open(download_path, "wb") as file:
        file.write(blob_client.download_blob().readall())
        print(f"Đã tải xuống {blob_name} từ Azure Blob Storage.")

# Tải mô hình và Label Encoder từ Azure Blob Storage
download_blob("label_encoder.pkl", "label_encoder.pkl")
download_blob("phobert_intent_classification.pth", "phobert_intent_classification.pth")

# Load trạng thái của mô hình đã huấn luyện
model.load_state_dict(torch.load("phobert_intent_classification.pth", map_location=device))
model.to(device)
model.eval()

# Load Label Encoder
label_encoder = joblib.load("label_encoder.pkl")

# Hàm dự đoán với độ tin cậy
def predict_intent_with_confidence(model, tokenizer, question, label_encoder, device):
    # Tiền xử lý câu hỏi
    inputs = tokenizer(question, return_tensors="pt", truncation=True, padding=True, max_length=128).to(device)
    
    with torch.no_grad():
        # Dự đoán và lấy logits
        outputs = model(**inputs)
        logits = outputs.logits
        
        # Chuyển logits thành xác suất bằng softmax
        probabilities = F.softmax(logits, dim=-1)
        
        # Lấy nhãn có xác suất cao nhất
        predicted_class = torch.argmax(probabilities, dim=-1).item()
        confidence = (probabilities[0, predicted_class].item())*100  # Xác suất của nhãn dự đoán
        
    # Giải mã nhãn về dạng văn bản
    intent = label_encoder.inverse_transform([predicted_class])[0]
    return intent, confidence

def handle_user_question_with_ai(user_input,storeId):
    intent,cf = predict_intent_with_confidence(model, tokenizer, user_input, label_encoder, device)
    print(intent)
    response = fetch_data(intent,user_input,storeId)
    print(cf)
    if cf>50:
        return response
    return None
    
    
@app.route('/intent-detection', methods=['POST'])
def intentDetection():
    data = request.json
    question = data['question']
    storeId = data['storeId']
    # print(question)
    response = handle_user_question_with_ai(question,storeId)
    return jsonify(response)
    

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)), debug=True)
