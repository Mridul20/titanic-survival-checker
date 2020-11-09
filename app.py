import numpy as np
import pickle
from flask import Flask,render_template,session,redirect,url_for
from flask_wtf import FlaskForm
from wtforms import StringField,SubmitField,RadioField,SelectField
from wtforms.fields.html5 import IntegerField
from wtforms.validators import DataRequired
import os 
from flask_sqlalchemy import SQLAlchemy



basedir = os.path.abspath(os.path.dirname(__file__))

app = Flask(__name__,template_folder = 'templates')
app.debug = True

model = pickle.load(open('model.pkl', 'rb'))

app.config['SECRET_KEY'] = "mysecretkey"
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' +os.path.join(basedir,'data.sqlite')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

class userinfo(db.Model):
    __tablename__ = 'userinfo'
    id = db.Column(db.Integer,primary_key = True)
    First_Name = db.Column(db.Text)
    Last_Name = db.Column(db.Text)
    PClass = db.Column(db.Text)
    Gender = db.Column(db.Text)
    Age = db.Column(db.Integer)
    Embarked = db.Column(db.Text)
    SibSp = db.Column(db.Integer)
    Parch = db.Column(db.Integer)

    def __init__(self,First_Name,Last_Name,PClass,Gender,Age,Embarked,SibSp,Parch):
        self.First_Name = First_Name
        self.Last_Name = Last_Name
        self.PClass = PClass
        self.Gender = Gender
        self.Age = Age
        self.Embarked = Embarked
        self.SibSp = SibSp
        self.Parch = Parch


class InfoForm(FlaskForm):

    First_Name = StringField("First Name:",validators = [DataRequired()])
    Last_Name = StringField("Last Name:",validators = [DataRequired()])
    PClass  = SelectField('Ticket Class:', choices=['1st','2nd','3rd'])
    Gender = SelectField('Gender:', choices = ['Male', 'Female'])
    Age = IntegerField('Age:')
    Embarked  = SelectField('Port of Embarkation:', choices=['Cherbourg','Queenstown','Southampton'])
    SibSp = IntegerField('No. of siblings on ship with you:', default=0)
    Parch = IntegerField('No. of parent on ship with you:', default=0)
    submit = SubmitField('Submit')

 
@app.route('/')
def index():
    return render_template('basic.html')

@app.route('/form',methods = ['GET','POST'])
def form():

    First_Name = False
    Last_Name = False
    PClass = False
    Gender = False
    Age = False
    Embarked = False
    SibSp = False
    Parch = False

    FORM = InfoForm()

    if FORM.validate_on_submit():
        session['First_Name'] = FORM.First_Name.data
        session['Last_Name'] = FORM.Last_Name.data
        session['PClass'] = FORM.PClass.data
        session['Gender'] = FORM.Gender.data
        session['Age'] = FORM.Age.data
        session['Embarked'] = FORM.Embarked.data
        session['SibSp'] = FORM.SibSp.data
        session['Parch'] = FORM.Parch.data
        usr = userinfo(First_Name = session['First_Name'],Last_Name = session['Last_Name'],PClass = session['PClass'],Gender=session['Gender'],Age=session['Age'],Embarked=session['Embarked'],SibSp=session['SibSp'],Parch = session['Parch'])
        db.session.add(usr)
        db.session.commit()
        if session['Gender'] == 'Male':
            session['Gender'] = 0
        else :
            session['Gender'] = 1
        if session['Embarked'] == 'Cherbourg':
            x = [int(session['PClass'][0]),1,0,int(session['Gender']),int(session['Age']),int(session['SibSp']),int(session['Parch'])]
        elif session['Embarked'] == 'Southampton':
            x = [int(session['PClass'][0]),0,1,int(session['Gender']),int(session['Age']),int(session['SibSp']),int(session['Parch'])]
        else:
            x = [int(session['PClass'][0]),0,0,int(session['Gender']),int(session['Age']),int(session['SibSp']),int(session['Parch'])]
        x = [np.array(x)]
        prediction = model.predict(x)   
        p_survive = model.predict_proba(x)[0][1]
        session['prob'] = p_survive
        if prediction == 1:
            return redirect(url_for('survived'))
        else:
            return redirect(url_for('notsurvived'))
    return render_template('form.html',form = FORM)

@app.route('/survived')
def survived():
    return render_template('survive.html')

@app.route('/notsurvived')
def notsurvived():
    return render_template('not_survive.html')

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run()


