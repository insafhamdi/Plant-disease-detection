from flask import Flask ,jsonify ,json
#from flask_login import LoginManager
#import pymongo 
#client = pymongo.MongoClient('localhost',27017)
#db1= client.livraison_system
from datetime import datetime , date 
#import isodate as iso 
#from bson import ObjectId
#from flask.json import JSONEncoder
from werkzeug.routing import BaseConverter


 

        
class ObjectIdConverter(BaseConverter):
    #def to_python(self, value):
        #return ObjectId(value)
    def to_url(self, value):
        return str(value)


def create_app():
    app = Flask(__name__)
    
    #app.json_encoder=MongoJSONEencoder
    app.url_map.converters['objectid']=ObjectIdConverter
     
    app.config.from_object('config')
    app.config['SECRET_KEY'] = b'8e7d3bf1c1bcb2b0ac747b8f64cd682516a0d392875107a8'
    
    #db.init_app(app)
    #login_manager = LoginManager()
    #login_manager.login_view = 'auth.login'
    #login_manager.init_app(app)

    #from .models import User 

    """     @login_manager.user_loader
    def load_user(user_id):
        # since the user_id is just the primary key of our user table, use it in the query for the user
        #return User.query.get(int(user_id))
        user_json = db1.users.find_one({'_id': ObjectId(user_id)})
        return User(user_json)
      """      
        
    # blueprint for auth routes in our app
    
    #from .auth import auth as auth_blueprint
    #app.register_blueprint(auth_blueprint)

    # blueprint for non-auth parts of app
    from .main import main as main_blueprint
    app.register_blueprint(main_blueprint)

    return app


 
 
