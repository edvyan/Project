## Running the Dash Application

### Install all the required packages using the command:
`pip install -r requirements.txt`

Follow these steps to run the Dash application locally on your machine:
### Step 1: Clone the repository to your local PC
git clone https://github.com/edvyan/Project.git

### Step 2: Download fine tuned model from below link, place it in app/sentiment_analysis/sentiment-model/
https://drive.google.com/drive/folders/18POzo4LBttTox_geu3acMwGlb_FxY8Ph?usp=sharing

### Step 3: Navigate to the app folder

### Step 4: Start the application
Open a terminal and type: python app.py

### Step 4: Access the Application
After the application has initialized, open a web browser and go to the following URL to access the Dash app:
http://127.0.0.1:8050/

### Step 5: Start a Conversation

With the Dash app open in your browser, users can start interacting with the chatbot by typing messages into the input box, pressing Enter or clicking the "Send" button to submit your message.

**Promote example:**
- Basic greetings (Hello!/How are you?)
- Tell me about Microsoft
- Show me TESLA
- I want to know about Apple stock

*Due to API limitations, we use local data to simulate API extraction. APIs can be implemented to replace local data. We have local data for Microsoft, Apple, TESLA* 

*Please note that the sentiment analysis is inaccurate due to finBERT not being fine-tuned on your local PC.*

### App Interface
Below is a screenshot of what the app interface looks like:
![image](https://github.com/edvyan/Project/assets/46171741/7ceec8d4-a8af-44cf-b282-9ac27283a203)


