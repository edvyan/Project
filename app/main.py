from flask import Flask, request, redirect, render_template, url_for, session
from flask_sqlalchemy import SQLAlchemy
from flask_bootstrap import Bootstrap
from flask_assets import Environment, Bundle
import csv

from bot import get_personalised_stock_info, generate_response, get_latest_stock_data

app = Flask(__name__)
app.secret_key = 'sunilkey'

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.SQLite'
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
    { 'sender': 'bot', 'message': 'Hello, how are you doing today?'},
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
    industry = db.Column(db.String(100), nullable=False)

    def to_dict(self):
        return {
            'id': self.id,
            'code': self.code,
            'name': self.name,
            'industry': self.industry
        }

    def __repr__(self):
        return f"Company('{self.code}', '{self.name}')"

class UserInterest(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    company_id = db.Column(db.Integer, db.ForeignKey('company.id'), nullable=False)

    def __repr__(self):
        return f"UserInterest"

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
    if 'user' in session:
        return redirect(url_for('index'))
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
        user=session['user']
        if request.method == 'POST':
            company_id = request.form['stocks']
            insert_stmt = UserInterest(user_id=user['id'], company_id=company_id)
            db.session.add(insert_stmt)
            db.session.commit()
        
        all_companies = Company.query.all()
        companies = [company.to_dict() for company in all_companies]
        
        user_companies = UserInterest.query.filter_by(user_id=user['id']).all()
        user_company_ids = [uc.company_id for uc in user_companies]
        
        user_companies_list = [company if (company['id'] in user_company_ids) else None for company in companies]

        companies = [ company for company in companies if company not in user_companies_list]
        
        return render_template('profile.html', user=user, companies=companies, user_companies_list=user_companies_list)
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

        all_companies = Company.query.all()
        companies = [company.to_dict() for company in all_companies]
        user_companies = UserInterest.query.filter_by(user_id=user['id']).all()
        user_company_ids = [uc.company_id for uc in user_companies]
        user_companies_list = [company if (company['id'] in user_company_ids) else None for company in companies]
        user_companies_list = [company for company in user_companies_list if company is not None]
        # companies = [ company for company in user_companies_list if company not in user_companies_list]
        personalised_news = [get_personalised_stock_info(user_company['name']) for user_company in user_companies_list]
        personalised_stocks = [{**get_latest_stock_data(user_company['code']), "Name":user_company['name'], "Symbol":user_company['code']} for user_company in user_companies_list if get_latest_stock_data(user_company['code']) is not None]
        # Render the template
        if request.method == 'POST':
            query = request.form['query']
            print(query)
            chat_messages.insert(0, {
                'sender': 'user',
                'message': query
            })
            response = "Test" #generate_response(query)
            chat_messages.insert(0, {
                'sender': 'bot',
                'message': response
            })
        return render_template('index.html', user=user, personalised_news=personalised_news, personalised_stocks=personalised_stocks, chat_messages=chat_messages)
    else:
        return redirect(url_for('login'))
    
@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('login'))

if __name__ == '__main__':
    with app.app_context():       
        # Create the database tables
        db.create_all()

        if (len(Company.query.all()) < 1):
            with open('../data/stock.csv', 'r') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    code = row['Symbol']
                    name = row['Company Name']
                    industry = row['Industry']
                    
                    existing_company = Company.query.filter_by(code=code).first()
                
                    if existing_company:
                        print(f"Company with code '{code}' already exists. Skipping...")
                    else:
                        company = Company(code=code, name=name, industry=industry)
                        db.session.add(company)
            db.session.commit()
        
    app.run(debug=True)
