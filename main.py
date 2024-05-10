from flask import Blueprint , render_template , session , redirect ,request , url_for,jsonify,flash
#from flask_login import login_required, current_user
#from . import db1
from werkzeug.utils import secure_filename
from .models import * 
import os
main = Blueprint('main', __name__)

def start_session( user):
    session['user']=user
    return jsonify(user),200


@main.route('/')
def index():
    return render_template('index.html')

@main.route('/profile')
def profile():
    return render_template('index.html')

@main.route('/resultat')
def resultat():
    return render_template('resultat.html')
@main.route('/analyser')
def analyser():
    return render_template('analyser.html')

@main.route('/profile' , methods=['POST'])
def profile_post():
    text=[]
    file = request.files['file']
    # Vérifier si le fichier est vide
    if file.filename == '':
        flash('Aucun fichier sélectionné')
        return redirect(url_for('main.index'))

    # Vérifier si le fichier est autorisé
    if file :
        # Nom de fichier sécurisé
        filename = secure_filename(file.filename)
        # Enregistrer le fichier sur le serveur
        file.save(os.path.join(filename))
        text=predict_single_image(filename)
        #return redirect(url_for('main.index', filename=filename))
        start_session( text)
        return render_template('resultat.html', prediction=text)

    
    
    #return redirect(url_for('main.index'))
    return render_template('resultat.html', prediction=text)


