import os
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "./coral-polymer-380506-adee1b5627bd.json"

import cv2

from google.api_core.client_options import ClientOptions
from google.cloud import documentai
import nltk

from nltk import ne_chunk, pos_tag, word_tokenize
from nltk.tree import Tree

import csv 
import re
import boto3
import numpy as np
import json
from flask import Flask
app = Flask(__name__)






project_id = 'coral-polymer-380506'
location = 'us' # Format is 'us' or 'eu'
processor_id = '51536eb2e8e1c33e' #  Create processor before running sample
mime_type = 'image/jpeg' # Refer to https://cloud.google.com/document-ai/docs/file-types for supported file types


@app.route('/process-test/<bucket_name>/<folder_id>')
def get_programming_language(bucket_name, folder_id): 
    # s3_client =boto3.client('s3')
    # s3_bucket_name='aissistbucket'
    s3 = boto3.resource('s3',
                        aws_access_key_id= 'AKIAQTVIRWADDDMGVZKC',
                        aws_secret_access_key='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')
                        
    my_bucket=s3.Bucket(bucket_name)
    bucket_list = []
    for file in my_bucket.objects.filter(Prefix=folder_id):
        file_name=file.key
        if file_name.find(".jpeg")!=-1:
            bucket_list.append(file.key)
    print(len(bucket_list))
    print(bucket_list)
    process_test(bucket_list,my_bucket)
    return str(my_bucket)

def quickstartContent(
    project_id, location, processor_id, image_content, mime_type):
  opts = ClientOptions(api_endpoint=f"us-documentai.googleapis.com")
  client = documentai.DocumentProcessorServiceClient(client_options=opts)
  name = client.processor_path(project_id, location, processor_id)

  raw_document = documentai.RawDocument(content=image_content, mime_type=mime_type)
  request = documentai.ProcessRequest(name=name, raw_document=raw_document)
  result = client.process_document(request=request)

  document = result.document

  print("The document contains the following text:")
  # print(document.text)
  return document.text




def process_test(bucket_list,my_bucket):
    for file in bucket_list:
        output = []
        img = my_bucket.Object(file).get().get('Body').read()
        nparray = cv2.imdecode(np.asarray(bytearray(img)), cv2.IMREAD_COLOR)
        image = nparray
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

        close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15,3))
        close = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, close_kernel, iterations=1)

        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,3))
        dilate = cv2.dilate(close, dilate_kernel, iterations=1)

        cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        for c in cnts:
            area = cv2.contourArea(c)
            if area > 800 and area < 15000:
                x,y,w,h = cv2.boundingRect(c)
                cv2.rectangle(image, (x, y), (x + w, y + h), (222,228,251), -1)

        success, encoded_image = cv2.imencode('.jpg', image)
        content = encoded_image.tobytes()
        text_content = quickstartContent(project_id, location, processor_id, content, mime_type)
        print(text_content)

        nltk_results = ne_chunk(pos_tag(word_tokenize(text_content)))
        for nltk_result in nltk_results:
            name = ''
            if type(nltk_result) == Tree and nltk_result.label() == "PERSON":
                for nltk_result_leaf in nltk_result.leaves():
                    name += nltk_result_leaf[0] + ' '
                print ('Type: ', nltk_result.label(), 'Name: ', name)
                break

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        # Apply Hough Circle Transform
        circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, dp=1, minDist=400, param1=200, param2=30, minRadius=20, maxRadius=50)

        # Create a blank mask
        mask = np.zeros(image.shape[:2], dtype=np.uint8)

        # Draw filled circles on the mask
        if circles is not None:
        circles = np.round(circles[0, :]).astype(int)

        for (x, y, r) in circles:
            cv2.circle(mask, (x, y), r, 255, -1)



        # Apply bitwise AND operation to retain only the selected circles in the original image
        result = cv2.bitwise_and(image, image, mask=mask)

        success, encoded_image = cv2.imencode('.jpg', result)
        content_2 = encoded_image.tobytes()

        client = vision.ImageAnnotatorClient()


        image = vision.Image(content=content_2)

        feature = vision.Feature(type_=vision.Feature.Type.TEXT_DETECTION)

        request = vision.AnnotateImageRequest(image=image, features=[feature])
        response = client.annotate_image(request=request)

        dictionary = {
        '1': 'NA',
        '2': 'NA',
        '3': 'NA',
        '4': 'NA',
        '5': 'NA',
        '6': 'NA',
        '7': 'NA',
        '8': 'NA'
        }


        single_alphabets = [annotation for annotation in response.text_annotations if re.match(r'^[A-Za-z]$', annotation.description)]


        startx = single_alphabets[0].bounding_poly.vertices[0].x
        starty = single_alphabets[0].bounding_poly.vertices[0].y
        description = single_alphabets[0].description
        average_y_threshold = 700


        i = 0
        index = 0

        print(startx)
        print(starty)
        print(description)

        items = list(dictionary.items())

        while index < len(dictionary):
        key, value = items[index]
        if(i >= len(single_alphabets)):
            break
        # for annotation in response.text_annotations:
        annotation = single_alphabets[i]
        x = annotation.bounding_poly.vertices[0].x
        y = annotation.bounding_poly.vertices[0].y
        if(abs(startx - x) <= 50):
            if(abs(starty - y) <= 1100):
                dictionary[key] = annotation.description
                i = i + 1
                starty = y
            else:
                dictionary[key] = ''
                starty = starty + average_y_threshold
        else:
            startx = x
            starty = single_alphabets[0].bounding_poly.vertices[0].y
            continue

        index = index + 1


        for key, value in dictionary.items():
        print(key, value)


        