import os
from flask import Flask, Response
from flask_restful import Api
from flask_cors import CORS
from api.ocr import InOcr
from api.foodOcr import FoodOcr

app = Flask(__name__)

CORS(app, resources={r"/*": {"origins": "*"}})
api = Api(app)

api.add_resource(InOcr, "/in-ocr")
api.add_resource(FoodOcr, "/food-ocr")

# api.add_resource(클래스, "도메인")

if __name__ == '__main__':
    # app.run(debug=True)
    app.run(host='127.0.0.1', port=5000)