import os
from flask import Flask, Request, redirect, url_for
from werkzeug import secure_filename

app = Flask(__name__)

WTF_CSRF_ENAMBLED = True
SECRET_KEY = 'you-will-never-guess'

UPLOAD_FOLDER = os.getcwd() + 'app/uploads'
ALLOWED_EXTENSIONS = set(['txt', 'csv'])

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 1024*1024

OPENID_PROVIDERS = [
    {'name': 'Yahoo', 'url': 'https://me.yahoo.com'},
    {'name': 'AOL', 'url': 'http://openid.aol.com/<username>'},
    {'name': 'Flickr', 'url': 'http://www.flickr.com/<username>'},
    {'name': 'MyOpenID', 'url': 'https://www.myopenid.com'}]


basedir = os.path.abspath(os.path.dirname(__file__))

SQLALCHEMY_DATABASE_URI = 'sqlite:///' + os.path.join(basedir, 'app.db')
SQLALCHEMY_MIGRATE_REPO = os.path.join(basedir, 'db_repository')
