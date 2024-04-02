# AI-Powered Financial Advisor and News Summarization System

## 1. Problem and Motivation
### Problem Statement
The modern financial market is highly dynamic, making it difficult for investors, both amateur and professional, to keep up-to-date with the latest news and stock performances. The overwhelming volume of financial news and data makes it challenging to extract meaningful insights efficiently.

The Finance Advisor project helps people deal with too much financial information and not enough time. Investors and financial experts often find it hard to keep up with all the news and data about stocks and markets. There's just so much information out there, and it's always changing. This can make it tough for people to figure out what's important and make smart decisions. So, we need a way to gather all this information, summarize it, and make it easier for people to understand. The Finance Advisor does this by letting users pick what they're interested in, like specific topics or companies. It then gives them a quick summary of the important stuff and offers advice based on that. This helps people make better decisions and manage risks more effectively when it comes to their money. Ultimately, the goal is to help people navigate the financial world more easily and succeed in their financial goals.

Regular people who invest their own money (individual investors) have a tough time keeping track of all the changes happening in the financial world. It takes a lot of time to search for important news articles from different websites, and these articles can be hard to understand if you're not a financial expert. Because this information is hard to find and understand, it's difficult for them to make smart decisions about their investments.

The "Finance Advisor" project solves this problem by creating a tool that's easy to use. This tool helps in three ways:

- It finds news articles related to specific companies or topics that you choose.
- It shortens these articles and makes them easier to read.
- It gives you basic advice about investing based on the news and how the stocks are doing.

### Motivation
The Finance Advisor project wants to help people deal with the overwhelming amount of financial news and data they face every day. We aim to create a platform where users can easily find the information they need to make smart financial decisions. By gathering and organizing financial news and data in one place, the Finance Advisor hopes to give users the tools and knowledge to understand what's happening in the financial world and make informed choices. They also want to make this information accessible to everyone, from beginners to experts, by making the platform easy to use. Overall, the goal of the Finance Advisor project is to make it easier for people to manage their money and investments confidently.


The "Finance Advisor" project tackles these challenges head-on by creating a user-friendly tool. This tool simplifies the process in three ways: firstly, by automatically finding news articles related to the user's chosen topics or companies. Secondly, by summarizing these articles into clear and concise pieces, making them easier to understand. Finally, by offering basic investment advice based on both the summarized news and historical stock performance data. By automating these tasks and making complex information more digestible, "Finance Advisor" empowers individual investors to save time, gain easier access to relevant information, and ultimately make well-informed choices about their investments.

Our aim is to develop a solution that simplifies this process by providing a personalized and summarized view of financial news and stock performance. This will enable users to make well-informed financial decisions quickly, enhancing their investment strategies.

## 2. Solution Requirements

1. #### User Requirements:
- Users can specify topics or companies of interest.
- The system should display the past 7 days' worth of news articles related to the user's query.
- Users should be able to access summaries of the news articles for quick comprehension.
- The system should display the past 7 days' worth of stock performance data for the user's chosen companies.
- The system should provide basic investment insights and advice based on the summarized news and stock data.

2. #### Technical Requirements
- APIs to fetch recent news and stock data.
- (NLP)Training and Modeling data for news summarization and insight generation.
- Application for integrated with APIs where users can interact

3. #### Functional Requirements:
- News API Integration: Integrate with a reliable financial news API to fetch relevant articles based on user-specified topics or companies.
- News Summarization Engine: Implement a news summarization algorithm to extract key points from retrieved articles and present them in a concise format.
- Stock Data API Integration: Integrate with a financial data API to retrieve historical stock performance data for chosen companies.
- Data Analysis and Insights Generation: Analyze the summarized news and stock data to generate basic investment insights and advice    tailored to the user's interests.
- User Interface: Design a user-friendly interface that allows users to easily specify their interests, view news summaries, and access stock data and insights.

4. #### Non-Functional Requirements:
- The system should retrieve news articles and stock data efficiently, with minimal lag time.
- The system should be scalable to accommodate a growing user base and increased data volume.
- The system should handle user data securely and comply with relevant data privacy regulations.
- The system should be highly available with minimal downtime.


## 3. Architecture (Framework)
1. **Data Collection Layer:** APIs for fetching news and stock data.
2. **Processing Layer:**
    - NLP Engine: For summarization and analysis of news articles.
    - Financial Analysis Module: For processing and interpreting stock data.
3. **Insight Generation Layer:** AI algorithms to derive insights and advice.
4. **Presentation Layer:** Front-end interface for user interaction.
5. **Database:** Storing user preferences and historical data.

## 4. Experimental Design (Methodology)
1. **API Integration:** Ensuring reliable and efficient data fetching.
2. **NLP Model Selection:** Testing different NLP models for effective summarization.
3. **User Feedback Loops:** Iteratively improve summaries and insights based on user feedback.
4. **Performance Metrics:** Evaluating system performance (accuracy, response time).
5. **Scalability Tests:** Ensuring system can handle increased loads.

## 5. Task Distribution
### Collaborative and Parallel Work Approach

#### Group 1:
- **API Integration and Backend Development:**
  - Focus on integrating APIs for data collection.
  - Develop backend logic for data processing and storage.
  - Collaborate with Team Member 3 for data needs of insight generation.

#### Groupr 2:
- **Front-End Development and User Experience:**
  - Design and implement the user interface.
  - Ensure seamless integration with the backend for data display.
  - Work closely with Team Member 1 for API integration on the front end.

#### Group 3:
- **NLP and Insight Generation Algorithms:**
  - Develop NLP models for news summarization.
  - Create algorithms for generating insights from processed data.
  - Collaborate with Team Members 1 and 2 to ensure data is appropriately processed and displayed.

### Shared Responsibilities:
- **Regular Meetings and Progress Reviews:**
  - All members should participate in regular meetings to discuss progress, challenges, and integration points.
- **Testing and Quality Assurance:**
  - Joint responsibility for testing the integrated system.
  - Collaborate on debugging and optimizing the application.
- **User Feedback and Iteration:**
  - Collect and analyze user feedback.
  - Work together to implement changes based on feedback.

## Conclusion
This project aims to bridge the gap between vast financial data and actionable insights for users. By leveraging AI and NLP, the system will provide personalized, summarized, and insightful financial advice, catering to the needs of a diverse range of investors.



# Summary of papers

## 1. Prompted LLMs as Chatbot Modules for Long Open-domain Conversation (Edvin Yang)
[Prompted LLMs as Chatbot Modules for Long Open-domain Conversation](https://aclanthology.org/2023.findings-acl.277) (Lee et al., Findings 2023)
### Summary
The document introduces a novel approach known as the Modular Prompted Chatbot (MPC) for developing high-quality conversational agents. This method leverages the capabilities of pre-trained large language models (LLMs) without requiring additional fine-tuning.

### Key Components
- **Modular Approach:** MPC utilizes pre-trained LLMs as individual modules, enhancing long-term consistency and flexibility in conversations.
- **Innovative Techniques:** The system incorporates advanced techniques like few-shot prompting, chain-of-thought (CoT), and external memory to improve performance.

### Main Findings
- **Performance Comparison:** According to human evaluations, MPC performs comparably to fine-tuned chatbot models in open-domain conversations.
- **Adaptability of Pre-trained LLMs:** The research emphasizes the ability of pre-trained LLMs to adapt to new tasks without the need for fine-tuning.

### Significance
The development of MPC underscores the importance of creating consistent and engaging chatbots for open-domain conversations, demonstrating the potential of using pre-trained LLMs in advanced conversational agents.

