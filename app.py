import sys
from os import walk
import imghdr
import csv
import argparse
import os
from flask import Flask, redirect, url_for, request, flash
from flask import render_template
from flask import send_file

import cv2
from model import CountRegressor, Resnet50FPN
from utils import MAPS, Scales, Transform, extract_features
from utils import visualize_output_and_save, select_exemplar_rois
from PIL import Image
import os
import torch
import argparse
import torch.optim as optim
from utils import MincountLoss, PerturbationLoss
from tqdm import tqdm



resnet50_conv = Resnet50FPN()
regressor = CountRegressor(6, pool='mean')
resnet50_conv.cuda()
regressor.cuda()
regressor.load_state_dict(torch.load("./data/pretrainedModels/FamNet_Save1.pth"))


resnet50_conv.eval()
regressor.eval()


INPUT_NAME = "./orange.jpg"
def process(INPUT_NAME,bboxes):
    image_name = os.path.basename(INPUT_NAME)
    image_name = os.path.splitext(image_name)[0]

    lines = [[1,2,3,4]]
    rects1 = list()
    for data in bboxes:
        y1 = int(data[0])
        x1 = int(data[1])
        y2 = int(data[2])
        x2 = int(data[3])
        rects1.append([y1, x1, y2, x2])

    print("Bounding boxes: ", end="")
    print(rects1)

    image = Image.open(INPUT_NAME)
    image.load()
    sample = {'image': image, 'lines_boxes': rects1}
    sample = Transform(sample)
    image, boxes = sample['image'], sample['boxes']

    image = image.cuda()
    boxes = boxes.cuda()    

    with torch.no_grad():
        features = extract_features(resnet50_conv, image.unsqueeze(0), boxes.unsqueeze(0), MAPS, Scales)


    features.required_grad = True
    #adapted_regressor = copy.deepcopy(regressor)
    adapted_regressor = regressor
    adapted_regressor.train()
    optimizer = optim.Adam(adapted_regressor.parameters(), lr=1e-7)

    pbar = tqdm(range(100))
    for step in pbar:
        optimizer.zero_grad()
        output = adapted_regressor(features)
        lCount = 1e-9 * MincountLoss(output, boxes, use_gpu=True)
        lPerturbation = 1e-4 * PerturbationLoss(output, boxes, sigma=8, use_gpu=True)
        Loss = lCount + lPerturbation
        # loss can become zero in some cases, where loss is a 0 valued scalar and not a tensor
        # So Perform gradient descent only for non zero cases
        if torch.is_tensor(Loss):
            Loss.backward()
            optimizer.step()

        pbar.set_description('Adaptation step: {:<3}, loss: {}, predicted-count: {:6.1f}'.format(step, Loss.item(), output.sum().item()))

    features.required_grad = False
    output = adapted_regressor(features)

    rslt_file = "{}/{}_out.png".format("./", image_name)
    visualize_output_and_save(image.detach().cpu(), output.detach().cpu(), boxes.cpu(), rslt_file)


app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0



from werkzeug.utils import secure_filename

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif', "webp"])

@app.route('/index')
def show_index():
    return render_template("index.html", user_image = "orange.jpg")

def allowed_file(filename):
    print(filename)
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
	
@app.route('/')
def upload_form():
	return render_template('upload.html')

@app.route('/', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        print('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        print('No image selected for uploading')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(".", filename))
        print('Image successfully uploaded and displayed below')
        print("sds")
        labels = app.config["LABELS"]
        labels=app.config["LABELS"]
        not_end = not(app.config["HEAD"] == len(app.config["FILES"]) - 1)
        print(not_end)
        app.config["FILES"] = [filename] + app.config["FILES"]
        return render_template('tagger.html', not_end=not_end, directory=".", image=filename, labels=labels, head=app.config["HEAD"] + 1, len=len(app.config["FILES"]))
    	#return render_template('upload.html', filename=filename)
    else:
        print('Allowed image types are -> png, jpg, jpeg, gif')
        return redirect(request.url)

@app.route('/display/<filename>')
def display_image(filename):
	print('display_image filename: ' + filename)
	return redirect(url_for('static', filename='./' + filename), code=301)

@app.route('/tagger')
def tagger():
    if (app.config["HEAD"] == len(app.config["FILES"])):
        return redirect(url_for('bye'))
    directory = app.config['IMAGES']
    image = app.config["FILES"][app.config["HEAD"]]
    labels = app.config["LABELS"]
    not_end = not(app.config["HEAD"] == len(app.config["FILES"]) - 1)
    print(not_end)
    return render_template('tagger.html', not_end=not_end, directory=directory, image=image, labels=labels, head=app.config["HEAD"] + 1, len=len(app.config["FILES"]))

@app.route('/next')
def next():
    image = app.config["FILES"][app.config["HEAD"]]
    app.config["HEAD"] = app.config["HEAD"] + 1
    input_bboxes = []
    with open(app.config["OUT"],'a') as f:
        for label in app.config["LABELS"]:
            f.write(image + "," +
            label["id"] + "," +
            label["name"] + "," +
            str(round(float(label["xMin"]))) + "," +
            str(round(float(label["xMax"]))) + "," +
            str(round(float(label["yMin"]))) + "," +
            str(round(float(label["yMax"]))) + "\n")
            input_bboxes.append([round(float(label["yMin"])),round(float(label["xMin"])),round(float(label["yMax"])),round(float(label["xMax"]))])
    app.config["LABELS"] = []
    if len(input_bboxes) != 0 :    
        process(app.config["FILES"][app.config["HEAD"]-1],input_bboxes)
        app.config["FILES"] = [app.config["FILES"][app.config["HEAD"]-1].split(".")[0] + "_out.png"] + app.config["FILES"]
        app.config["HEAD"] = 0
    return redirect(url_for('tagger'))

@app.route("/bye")
def bye():
    return send_file("taf.gif", mimetype='image/gif')

@app.route('/add/<id>')
def add(id):
    xMin = request.args.get("xMin")
    xMax = request.args.get("xMax")
    yMin = request.args.get("yMin")
    yMax = request.args.get("yMax")
    app.config["LABELS"].append({"id":id, "name":"", "xMin":xMin, "xMax":xMax, "yMin":yMin, "yMax":yMax})
    return redirect(url_for('tagger'))

@app.route('/remove/<id>')
def remove(id):
    index = int(id) - 1
    del app.config["LABELS"][index]
    for label in app.config["LABELS"][index:]:
        label["id"] = str(int(label["id"]) - 1)
    return redirect(url_for('tagger'))

@app.route('/label/<id>')
def label(id):
    name = request.args.get("name")
    app.config["LABELS"][int(id) - 1]["name"] = name
    return redirect(url_for('tagger'))

# @app.route('/prev')
# def prev():
#     app.config["HEAD"] = app.config["HEAD"] - 1
#     return redirect(url_for('tagger'))

@app.route('/image/<f>')
def images(f):
    images = app.config['IMAGES']
    return send_file(images + f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('dir', type=str, help='specify the images directory')
    parser.add_argument("--out")
    args = parser.parse_args()
    directory = args.dir
    if directory[len(directory) - 1] != "/":
         directory += "/"
    app.config["IMAGES"] = directory
    app.config["LABELS"] = []
    files = None
    for (dirpath, dirnames, filenames) in walk(app.config["IMAGES"]):
        files = filenames
        break
    if files == None:
        print("No files")
        exit()
    app.config["FILES"] = files
    app.config["HEAD"] = 0
    if args.out == None:
        app.config["OUT"] = "out.csv"
    else:
        app.config["OUT"] = args.out
    print(files)
    with open("out.csv",'w') as f:
        f.write("image,id,name,xMin,xMax,yMin,yMax\n")
    app.run(debug="True",host="0.0.0.0")
