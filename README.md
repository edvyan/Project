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

#### Edvin and Tairo
- **API Integration and Backend Development:**
  - Focus on integrating APIs for data collection.
  - Develop backend logic for data processing and storage.
  - Collaborate with Team Member 3 for data needs of insight generation.

#### Sunil
- **Front-End Development and User Experience:**
  - Design and implement the user interface.
  - Ensure seamless integration with the backend for data display.
  - Work closely with Team Member 1 for API integration on the front end.

#### Nitesh and Stabya
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

## 1. Prompted LLMs as Chatbot Modules for Long Open-domain Conversation (Edvin)
[Prompted LLMs as Chatbot Modules for Long Open-domain Conversation](https://aclanthology.org/2023.findings-acl.277) (Lee et al., Findings 2023)
### Summary
The document introduces a novel approach known as the Modular Prompted Chatbot (MPC) for developing high-quality conversational agents. This method leverages the capabilities of pre-trained large language models (LLMs) without requiring additional fine-tuning.

### Methods
- **Modular Approach:** MPC utilizes pre-trained LLMs as individual modules, enhancing long-term consistency and flexibility in conversations.
- **Innovative Techniques:** The system incorporates advanced techniques like few-shot prompting, chain-of-thought (CoT), and external memory to improve performance.

### Main Findings
- **Performance Comparison:** According to human evaluations, MPC performs comparably to fine-tuned chatbot models in open-domain conversations.
- **Adaptability of Pre-trained LLMs:** The research emphasizes the ability of pre-trained LLMs to adapt to new tasks without the need for fine-tuning.

### Conclusion
The development of MPC underscores the importance of creating consistent and engaging chatbots for open-domain conversations, demonstrating the potential of using pre-trained LLMs in advanced conversational agents.

## 2. An Exploration of Automatic Text Summarization of Financial Reports (Edvin)
[An Exploration of Automatic Text Summarization of Financial Reports](https://aclanthology.org/2021.finnlp-1.1.pdf) (Abdaljalil & Bouamor, FinNLP 2021)

### Summary
The document introduces a noval approach which simplifies the processing of extensive financial documents and can aid investment decisions through efficient summarization.

### Methods
- NLP, machine learning, and deep learning techniques.
- Sentence-based and section-based summarization.
- Use of BERT for text encoding.
- Unsupervised clustering to group sections.

### Main Findings
- Evaluation using a dataset of British firms' annual reports.
- Best model achieves a ROUGE-L score of 36%.

### Conclusion
- Effective methods for financial narrative summarization.
- Potential for further development in section identification and extraction.



## 3. FACTKB: Generalizable Factuality Evaluation using Language Models Enhanced with Factual Knowledge (Nitesh)
[FACTKB: Generalizable Factuality Evaluation using Language Models Enhanced with Factual Knowledge](https://arxiv.org/pdf/2305.08281v2.pdf)

### Summary 
This paper discusses a program called FACTKB, a tool that evaluates if automatically created summaries are factually correct. At the moment, the tools we have to do this task still make mistakes, especially when it comes to identifying entities (specific items, objects, or people) and relations (ways these entities relate to each other) in new areas or topics.

### FACTKB methodology
The FACTKB methodology is designed to enhance the understanding of 'facts' or 'truths' in language models (LMs). Here's a simple explanation of its processes:
`Factuality Pretraining`: This process works with knowledge bases (KBs), which are rich sources of facts. The goal is to use KBs as 'fact teachers' to improve the language model's understanding of entities and relations.
It utilizes three strategies:
`Strategy 1 - Entity Wiki`: This strategy works with the practice of predicting missing connections in KBs based on available facts. The language model is trained to anticipate masked entities or relations in the facts provided by the KBs. This trains LMs to infer facts from surrounding information and punishes unsupported claims about entities and relations.
`Strategy 2 - Evidence Extraction`: This strategy is about enhancing the model’s ability to judge facts based on adequate evidence. Here, a triple (ei, rk, ej) is selected randomly from the KB. The first paragraph of the Wikipedia description of ei is used as auxiliary knowledge. Then, a sentence is formed using these two, creating a corpus of triples mixed with auxiliary knowledge. This corpus aids FACTKB in selecting evidence from documents to support its factuality evaluation.
Strategy 3 - Knowledge Walk: This strategy is meant to improve the understanding of multi-hop claims. The idea is to have a randomly picked entity and selected further from its direct neighborhood, forming one-hop triple {e(0), r(0,1), e(1)}. This process is repeated several times, creating a corpus used for factuality pretraining.
LM Training: After the factuality pretraining, the language models are then trained using the proposed methods, refining the fact-enhanced LM on a factuality error detection dataset.
Through these methods, FACTKB is expected to improve in understanding and evaluating the factuality of entities and relations in a document.

### FACTKB training 
The FACTKB training process begins with initializing FACTKB with encoder-based Language Models (LMs). Each of the three factuality pretraining strategies is then individually applied to FACTKB in a process involving the masked language modeling approach. The goal is to evaluate the effectiveness of each strategy. This results in LMs that are enhanced with the ability to better represent facts, entities, and relationships.

### Data and Experiment
The experimentation begins with training the model using a dataset from YAGO, a knowledge base built from Wikidata for factuality pretraining. The refining of the model, however, is done using the FactCollect dataset which collects human annotations concerning factual errors from various sources such as CNN, Daily Mail, and BBC, and consolidates them into a single dataset. For each pair of summary and article, a FACTUAL or NON-FACTUAL label is assigned.
The model is configured with some specific resource limits and parameters: a corpus size of 100,000, a masking probability of 15%, and a knowledge walk length of 5. The model is then pretrained for 5 epochs and fine-tuned for a maximum of 50 epochs. Three types of factuality pretraining are conducted.
Then comes the evaluation stage which consists of in-domain and out-of-domain evaluations. The in-domain evaluation is focused on the news media. The FactCollect dataset and the FRANK benchmark serve as the basis for this evaluation. The FRANK benchmark provides human judgments on the factual consistency of model-generated summaries, categorizing various factual errors.
However, as summary systems are used across various domains (news, social media, scientific literature, etc.), the model is also evaluated for its ability to provide reliable factuality scores across these varied domains. For this, three datasets from the scientific literature domain are used: CovidFact, HealthVer, and SciFact. CovidFact collects claims from a subreddit and verifies them against relevant scientific literature and Google search results. HealthVer consists of claims from TREC-COVID and verified against the CORD-19 corpus. SciFact includes claims from biomedical literature and verifies them against the abstract of the cited paper.
Finally, the performance of FACTKB is compared with existing factuality evaluation models like QAGS, QUALS, DAE, FalseSum, SummaC, FactCC, and FactGraph to understand its relative effectiveness.

### Result:
The existing models only performed slightly better than random factuality scores on these scientific literature datasets, suggesting their limitations in generalizing to other domains. The results shows that FACTKB exceeded performance expectations in both in-domain and out-of-domain settings, validating its effectiveness and robustness in different applications.

### Analysis:
The analysis showed that FACTKB has made significant improvements in factual error detection, can work with various language models and knowledge bases, and is a more straightforward and lightweight method for factuality evaluation. The performance of FACTKB can be influenced by different parameters such as corpus size, number of pretraining epochs, and knowledge walk length.

### Conclusion:
Detailed experiments show that FACTKB performs better than existing methods in evaluating factuality within the same domain of news media and across different domains, such as scientific literature. It also aligns well with human judgments on truthfulness and is especially proficient in detecting semantic frame errors, which are types of factual errors related to entities and relations.
Thus, FACTKB serves as a user-friendly and adaptable metric for factuality, which will aid research on producing factually consistent summaries.


## 4. Enhancing Semi-Supervised Learning for Extractive Summarization with an LLM-based pseudolabeler (Nitesh)
[FACTKB: Generalizable Factuality Evaluation using Language Models Enhanced with Factual Knowledge](https://arxiv.org/pdf/2311.09559.pdf)

### Summary
The paper addresses the challenge of creating brief summaries of text, particularly when there is limited labeled data available. The researchers propose a technique where they use a language model called GPT-4 to select the best pseudolabels or artificially created labels for summarization. They evaluate this technique on three types of datasets: TweetSumm, WikiHow, and ArXiv/PubMed. The results show that the use of GPT-4 improves summary quality significantly across these datasets, proving that the method is highly effective. They also found that this method requires fewer unlabeled examples to work effectively.

### Methodology
Here, how a self-supervised summarization model is created by applying a standard teacher-student training process and using GPT-4 for pseudolabel generation is explained. Firstly, an initial 'teacher' model is trained using limited labeled data. This teacher then generated pseudolabels (artificial labels) for the unlabeled dataset. Top 50 pseudolabels are selected based on the teacher’s confidence level. To quantify the teacher model's confidence in a given pseudolabel, the average predicted probability of each sentence is calculated in the pseudolabel being part of the summary.
Next, using GPT-4, a numerical rating to each of these top 50 pseudolabels are provided based on certain criteria. These criteria included the summary being concise, covering key points from the original text, and being an extractive summary meaning it should directly use sentences from the input conversation without any modifications.
Finally, the top 5 pseudolabels with the highest GPT-4 score are selected, then used GPT-4 to generate new pseudolabels for unlabeled examples which were then added to the training data for the next cycle. The same steps were then repeated in each cycle to continually refine the model.

### Experimental Setup
In the experimental setup, three datasets were used: 
  - `TweetSumm`, which contains customer service chat data 
  - `Wikihow`, which contains articles with corresponding summaries
  - `ArXiv/PubMed`, a collection of scientific articles with their abstracts as the summaries.
Initially models were trained on a small subset of labelled examples from each dataset, and the rest of the data was treated as unlabeled. The trained model were to generate pseudolabels (artificial labels) for the unlabeled examples. 50 pseudolabels were selected based on the model's confidence in its labelling.
Then, GPT-4 was used to give a score to each pseudolabel by considering several criteria such as the conciseness of the summary, coverage of key points from the text, and whether it is an extractive summary (directly using sentences from the conversation without modification). The top 5 pseudolabels with the highest GPT-4 scores were added to the training set for the next cycle.
This process was repeated over multiple training cycles until they had a final set of 300 examples (50 labeled and 250 pseudolabels). The models' performance was evaluated using ROUGE scores, which consider the overlap of n-grams (continuous sequence of n items) between the predicted and actual summaries.
Finally, the researchers compared their approach with different baselines including a teacher-student learning framework, self-supervised pre-training, the original PreSumm model, and a random model selection approach.

### Results:
Different models and strategies were compared  and it was found that when more labels (or data) are used for training, models produce better summaries. However, this improvement was not so noticeable in WikiHow and ArXiv/PubMed datasets because the model's labels were created using a matching technique that might not be the best way to create labels.
Among the semi-supervised models (which learn from both labelled and unlabeled data), all selection strategies they tested did better than a model trained with random pseudolabels (artificial labels). They observed that using GPT-4 to rate pseudolabels improved the performance for all datasets, suggesting that the summarization model can generate some high-quality labels and that GPT-4 is beneficial for determining those high-quality labels.
It was concluded that the performance of their model, PreSumm, aligns poorly with the actual distribution of the data, possibly due to having only a small labelled dataset. Hence, depending on GPT-4 to rate generated pseudolabels can be a beneficial strategy, especially in situations where there is little labelled data available.


## 5. Comparative Analysis of Business Machine Learning in Making Effective Financial Decisions Using Structural Equation Model (SEM) 
The research paper titled "Comparative Analysis of Business Machine Learning in Effective Financial Decision Making Using Structural Equation Model (SEM)" explores the role of machine learning (ML) in improving financial decision making in organization s It focuses on ML strategies used in various economies, including risk management, forecasting, and customer communications, focusing on the potential for reducing costs, improving efficiency, and improving decision-making

The article focuses on ML in finance, particularly in algorithmic trading, risk management, and process automation. The ML algorithm generates customized reports based on available data, enabling fast and informed decision making. The study also includes the use of natural language processing (NLP) to analyze and integrate large amounts of data for market research purposes. ML is seen as a valuable tool for analyzing economic data, forecasting market trends, and supporting decision making in areas such as regression analysis and classification

Overall, the research article highlights the important role of machine learning in investment decision-making. It uses ML to optimize business processes, manage risks, and connect customers more effectively. By leveraging large amounts of data, ML enables organizations to make informed decisions, reduce costs, and improve overall financial performance.

## 6 








## References
Sujith, A.V.L.N. et al. (2022) A comparative analysis of business machine learning in making effective financial decisions using structural equation model (SEM), Journal of Food Quality. Available at: https://www.hindawi.com/journals/jfq/2022/6382839/ (Accessed: 03 April 2024). 








