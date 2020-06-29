from flask import Flask
from flask import render_template, flash, redirect
from app import app
from .forms import LoginForm
import threading
import subprocess
import uuid



@app.route('/')
@app.route('/index', methods = ['GET', 'POST'])
def index():
    return render_template('template\index.html',
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
    return render_templacte('processing.html', id=id)

@app.route('/is_done')
def is_done():
    id = request .args.get('id', None)
    if id not in background_scripts:
        abort(404)
    return jsonify(done=background_scripts[id])
