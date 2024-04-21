from flask import Flask, request, redirect, render_template, url_for, session
from flask_sqlalchemy import SQLAlchemy
from flask_bootstrap import Bootstrap
from flask_assets import Environment, Bundle

from bot import generate_response

app = Flask(__name__)
app.secret_key = 'sunilkey'

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'
Bootstrap(app)
assets = Environment(app)
db = SQLAlchemy(app)

scss = Bundle('login.scss', filters='scss', output='gen/main.css')
assets.register('scss_all', scss)

dashboard = Bundle('dashboard.scss', filters='scss', output='gen/dashboard.css')
assets.register('dashboard', dashboard)

profile = Bundle('profile.scss', filters='scss', output='gen/profile.css')
assets.register('profile', profile)

chat_messages = [
    { 'sender': 'user', 'message': 'I am doing fine'},
    { 'sender': 'bot', 'message': 'Hellow how are you doing today?'},
]

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    firstname = db.Column(db.String(50), nullable=False)
    lastname = db.Column(db.String(50), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(60), nullable=False)

    def to_dict(self):
        return {
            'id': self.id,
            'firstname': self.firstname,
            'lastname': self.lastname,
            'email': self.email
        }

    def __repr__(self):
        return f"User('{self.firstname}', '{self.lastname}', '{self.email}')"

# Define the Company model
class Company(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    code = db.Column(db.String(10), unique=True, nullable=False)
    name = db.Column(db.String(100), nullable=False)

    def __repr__(self):
        return f"Company('{self.code}', '{self.name}')"

user_company = db.Table('user_company',
    db.Column('user_id', db.Integer, db.ForeignKey('user.id'), primary_key=True),
    db.Column('company_id', db.Integer, db.ForeignKey('company.id'), primary_key=True)
)

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        # Get form data
        firstname = request.form['firstname']
        lastname = request.form['lastname']
        email = request.form['email']
        password = request.form['password']

        # Create new user object
        new_user = User(firstname=firstname, lastname=lastname, email=email, password=password)

        # Add user to database
        db.session.add(new_user)
        db.session.commit()

        user = User.query.filter_by(email=email).first()

        if user and user.password == password:
            # Save user object in session
            session['user'] = user.to_dict()  # Assuming to_dict() method is defined in the User model

            # Redirect to index page
            return redirect(url_for('index'))
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        # Get form data
        email = request.form['email']
        password = request.form['password']

        # Check if user exists in database
        user = User.query.filter_by(email=email).first()

        if user and user.password == password:
            # Save user object in session
            session['user'] = user.to_dict()  # Assuming to_dict() method is defined in the User model

            # Redirect to index page
            return redirect(url_for('index'))
        else:
            return render_template('login.html', error="Invalid email or password")
    return render_template('login.html')

@app.route('/profile', methods=['GET', 'POST'])
def profile():
    if 'user' in session:
        return render_template('profile.html', user=session['user'])
    else:
        return redirect(url_for('login'))

@app.route('/', methods=['GET', 'POST'])
@app.route('/index', methods=['GET', 'POST'])
def index():
    # Check if user is logged in
    if 'user' in session:
        user = session['user']
        # Access user attributes as needed
        firstname = user['firstname']
        lastname = user['lastname']
        # Render the template
        if request.method == 'POST':
            query = request.form['query']
            print(query)
            chat_messages.insert(0, {
                'sender': 'user',
                'message': query
            })
            response = generate_response(query)
            chat_messages.insert(0, {
                'sender': 'bot',
                'message': response
            })
        return render_template('index.html', user=user, chat_messages=chat_messages)
    else:
        return redirect(url_for('login'))

if __name__ == '__main__':
    with app.app_context():       
        # Create the database tables
        db.create_all()
        
    app.run(debug=True)
