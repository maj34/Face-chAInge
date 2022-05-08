from flask import Flask
from flask import Blueprint, request
from flask import send_from_directory
from flask.templating import render_template
import os, json
import inference

bp = Blueprint('main', __name__, url_prefix='/')

@bp.route('/')
def main():
    return render_template('main.html')

@bp.route('/index')
def index():
    return render_template('index.html')


@bp.route('/imageupload', methods=['POST'])
def image_submit():
    if request.method=='POST':
        file = request.files['file']
        if file.filename == None or file.filename == "":
            return render_template('index.html')
        file_name = os.path.join("images", file.filename)
        save_dir = os.path.join("static", file_name)
        file.save(save_dir)
        bounding_box = {"bb":inference.detection(save_dir)}
        return render_template("submit.html", image_name=save_dir, data=json.dumps(bounding_box))
        
        
@bp.route('/result', methods=['GET','POST'])
def face_swap():
    if request.method=='POST':
        tmp = json.loads(request.data)
        inference.face_swap(tmp['origin_path'], tmp['selection'])
        return "OK"
    else:
        param = request.args.to_dict()
        if len(param) == 0:
            return "ERROR"
        splitted = param['image'].split(".")
        return render_template("result.html", img_path="/static/images/" + (".".join(splitted[:-1]) if len(splitted) > 2 else splitted[0]) + "_result." + splitted[-1])


app = Flask(__name__)
app.register_blueprint(bp)
app.run("127.0.0.1", port="4000", debug=True)