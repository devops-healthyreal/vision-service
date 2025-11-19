import os
from flask import request
from flask_restful import Resource
import numpy as np
from google.cloud import vision
from google.oauth2 import service_account
import re
from PIL import Image
from dotenv import load_dotenv