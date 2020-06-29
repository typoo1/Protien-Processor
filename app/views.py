import threading
import subprocess
import uuid
import sys
import os
import time
from flask import Flask
from flask import render_template, url_for, abort, jsonify, request, flash, redirect, request, url_for, send_from_directory
from flask_wtf import Form
from flask_wtf.file import FileField
from wtforms import StringField
from wtforms.validators import DataRequired
from werkzeug import secure_filename
from app import app
import S230EAv10Part1
import S230EAv10Part2

app.config['UPLOAD_FOLDER'] = os.getcwd() + '/app/uploads/'

class UForm(Form):
    file = StringField('File')
    csvfile = FileField('Your CSV File')

background_scripts = {}

def run_script(id):
    subprocess.call(['python.exe', 'C:/Users/TyeG/CS Project/app/S230EAv10Part1.py'])
    background_scripts[id] = True

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

@app.route('/', methods = ['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        guan = request.files['guan']
        if file:
            filename = secure_filename(file.filename)
            guanfilename = secure_filename(guan.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            guan.save(os.path.join(app.config['UPLOAD_FOLDER'], guanfilename))
            S230EAv10Part1.run(filename, guanfilename)
            return redirect(url_for('display'))

    return '''
     <!doctype html>
     <title>Upload new File</title>
     <h1>Upload new File</h1>
     <form action="" method=post enctype=multipart/form-data>
     <h2>Input file</h2>
     <p><input type=file name=file> <br>
     <h2>Guan file</h2>
     <p><input type=file name=guan>
     <input type=submit value=Upload>
     </form>
    '''



@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
    
@app.route('/index', methods = ['GET', 'POST'])
def index():
    return render_template('index.html',
                           title='home',)

@app.route('/login', methods = ['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        flash('Login requested for OpenID="%s", remember_me=%s' %
              (form.openid.data, str(form.remember_me.data)))
        return redirect('/index')
    return render_template('login.html',
                           title='Sign In',
                           form=form,
                           providers=app.config['OPENID_PROVIDERS'])

@app.route('/generate')
def generate():
    id = str(uuid.uuid4())
    background_scripts[id] = False
    threading.Thread(target=lambda: run_script(id)).start()
    return render_template('processing.html', id=id)

@app.route('/is_done')
def is_done():
    time.sleep(5)
    id = request.args.get('id', None)
    if id not in background_scripts:
        S230EAv10Part1.run()
        abort(404)
    return redirect(url_for('display'))

@app.route('/display', methods=['GET', 'POST'])
def display():
    if request.method == 'POST':
        keep = request.form.getlist('keep')
        S230EAv10Part2.run(keep)
        return redirect(url_for('finish'))
		
    return'''
    <!doctype html>
      <title>Choose results to keep</title>
     <h1>Choose results to keep</h1>
     <img src="/static/unfoldPlot_0.png" alt="unfoldPlot_0" style="width:719px;height:523px;">
     <img src="/static/unfoldPlot_1.png" alt="unfoldPlot_1" style="width:719px;height:523px;">
     <img src="/static/unfoldPlot_2.png" alt="unfoldPlot_2" style="width:719px;height:523px;">
     <img src="/static/unfoldPlot_3.png" alt="unfoldPlot_3" style="width:719px;height:522px;">
     <img src="/static/unfoldCorrelation.png" alt="unfoldPlot_cross" style="width:719px;height:522px;">
     <img src="/static/allUnfolds.png" alt="unfoldPlot_comb" style="width:719px;height:522px;">
     <form action="" method=post enctype=multipart/form-data>
         <input type="checkbox" name="keep" value="p1">Keep Plot 1
         <input type="checkbox" name="keep" value="p2">Keep Plot 2<br>
         <input type="checkbox" name="keep" value="p3">Keep Plot 3
         <input type="checkbox" name="keep" value="p4">Keep Plot 4<br>
         <input type=submit value=Accept>
     </form>
    '''
@app.route('/finish')
def finish():
    return'''
    <!doctype html>
      <title>Choose results to keep</title>
     <a href="static/result_fit.csv"> result fit</a> <br>
     <a href="static/result_plot.pdf" download> result plot</a>
    '''
