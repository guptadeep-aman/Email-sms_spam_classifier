# EMAIL-SMS-SPAM-CLASSIFIER USING MACHINE LEARNING
The main aim of this project is to combine multiple services and open-source tools to make a system that will classify whether the e-mails and sms are spam or non-spams.

# Introduction
In the modern world where digitization is everywhere, SMS has become one of the most vital forms of communication, unlike other chatting-based messaging systems like Facebook, WhatsApp, etc., SMS does not require an active internet connection at all. As we all know Hackers / Spammer tries to intrude into Mobile Computing Device, and SMS support for mobile devices had become vulnerable, as attacker tries to intrude into the system by sending an unwanted link, which on clicking those link the attacker can gain remote access over the mobile computing device. So, to identify those messages Authors have developed a system that will identify such malicious messages and will identify whether or not the message is SPAM or HAM (malicious or not malicious). The authors have created a dictionary using the TF-IDF Vectorizer algorithm, which includes all the features of words a SPAM SMS possess, based on the content of the text and referring I to the dictionary of the system and further it will be classifying the SMS and e-mail as spam or ham.

SMS is one of the most effective forms of communication. It is based on a cellular communication system and only your mobile phone needs to be in the coverage area of the network to send and receive messages. 
Most people use this service for communication.
Various organizations using SMS to communicate with their clients, banks and other government agencies also use his SMS to communicate. 
Many business organizations also use this service for advertising purposes. SMS therefore plays an important role as this framework does not require any active Internet connection. 
The widespread use of SMS makes it one of the favorite places of hackers and spammers. Hackers can easily compromise someone else's cell phone by simply sharing or broadcasting a malicious link to their users. When an end-user clicks on a link or message sent by a hacker/spammer, her mobile device is automatically compromised. Once a hacker has control of your system, you can learn the rest of the ways hackers can exploit it. Therefore, limiting the content that end users receive has become very important. Therefore, we need a system to let end users know if a received message is spam. Non-spam messages he called HAM. 
We identified the above problems and issues and developed a system that can detect whether it is spam or HAM from the content of a message using machine learning techniques. In this section, the author gave a brief overview of machine learning. Different types of machine learning and the techniques the authors used to develop machine learning models.

Each point in this article is considered in the context of a large social app that spans many countries. 
Users are offered a level of free SMS messaging to any number and have proven to be an excellent service in areas with very limited internet connectivity and very high mobile data costs.
This naturally attracted the attention of scammers and scammers who found opportunities to make shady transactions very cheaply and led to many attempts to exploit the platform. This model and related procedures were developed in response to this situation to maintain the quality of user experience within the app.

Machine Learning: -  
Machine learning is an interesting field because it covers important parts of different disciplines: statistics, artificial intelligence theory, data 
analysis, and numerical methods. Machine learning can be defined as the semi-automatic extraction of knowledge from datasets or data. 
Let's split the definition into three components. 
i)	First, machine learning always starts with data, with the goal of extracting insights from the user's data or dataset. 
ii)	Second, machine learning involves some degree of automation rather than manually trying to extract insights from data. 
iii)	Finally, machine learning is not fully automated. H. 

A successful process requires human intervention to make many intelligent decisions. Simply put, machine learning is an application that allows you to improve your predictions with iterations and with experience. The process by which an application improves with experience is, of course, called training. 
Larger iterations may be required to gradually improve results. During training, data is passed to a machine learning algorithm that improves 
internal representations and numerical parameters when discrepancies 
or training errors occur. The goal of this stage is to minimize the cost and error functions or adjust the internal weights of the algorithm to maximize the probability. As the algorithm improves in accuracy, it is called learning. Once the results are sufficiently accurate (also known as scoring), machine learning applications can be used to solve the problems they were designed to solve.


What is spam and why should it be prevented?
Spam is unsolicited and unwanted messages sent electronically that may be 
malicious in content. I have. Email spam is sent and received over the
Internet, while SMS spam is typically sent over cellular networks. A user who sends spam is called a "spammer". SMS messages are usually very cheap (if not free) for users to send, which makes them attractive for illegal exploits. To make matters worse, SMS is almost always considered a more secure and reliable form of communication than other sources. E.g. Email.
The dangers of spam messages to users are numerous. Unwanted ads, disclosure of personal information, falling victim to fraud or financial 
schemes, being directed to malware or phishing websites, or being unknowingly exposed to inappropriate content. For network operators, this leads to increased spam message operational costs.
In this case, spam is annoying to users, negatively impacting quality of service, and damaging your brand. This can lead to complaints, low ratings and even loss of users, not to mention users getting scammed.


Differences with Email & Spam: - 
The following table summarizes the main differences between spam in email and SMS.

Spammer’s behavior: - 
Spammers attempt to test an operator's anti-spam infrastructure by sending varying amounts of spam to determine if volume barriers are in place. Using multiple numbers to send messages is very common and we rule out number blocking as an anti-spam strategy. requires some form of content-based filtering that
In-Service Spam Filtering Status At the beginning of the project, the only anti-spam measure was to block users whose number of SMS sent exceeded her daily and hourly thresholds. At that time, there was no content-based filtering or consideration based on the user's metadata. 
It was a rules-based system, very easy to circumvent, and used very little data.

# STEPS INVOLVED IN BUILDING THE PROJECT: - 

1. Data cleaning
2. Exploratory Data Analysis
3. Text Preprocessing
4. Model building
5. Evaluation of the model on the basis of accuracy
6. Improvement in the model
7. Website

- Data Cleaning: This is the first step that involves the importing of the dataset (for this model we have chosen it from the kaggle). The aim of information cleaning right here is to find the proper manner to rectify high-satisfactory issues like eliminating bad data, and filling in missing data to form the information efficient for the model.

- Exploratory Data Analysis: This is the second step in which EDA has been conducted on the 2 Columns and exploring and gaining more knowledge about the data. 
This dataset initially contains 87.35% of Ham and 12.63 Spam Data which is technically imbalanced for the model.

- Text Preprocessing: This step involves the following transformations.
•	Lower case
•	Tokenization
•	Removing special characters
•	Removing stop words and punctuation
•	Stemming (words with same meaning)

- Model building:  Email spam detection is a classification problem. Some algorithms  such as Naive Bayes Classifier and Decision Trees are good for spam detection. Algorithms such as KNN, linear regression, etc. do not perform well in practice due to their inherent drawbacks such as the curse of dimensionality.


# ML Algorithms: Project is trained with multiple models and compares their accuracy with each other.

•	Naïve Bayes with TF-IDF approach:  Naive Bayes is the simplest 
classification algorithm (quick to create and regularly used for spam detection). This is a common text classification problem that uses word frequency as a feature to determine whether a document belongs to one category or another (e.g., spam or legal, sports or politics, etc.).

•	Decision Trees Decision trees are used for classification and regression. Decision analysis provides a visual and explicit representation of decisions and decisions using decision trees. The decision to make strategic splits greatly affects the accuracy of the tree. Decision criteria are different for classification trees and regression trees. 
Information theory has a measure that defines this degree of confusion in a system known as entropy. If the sample is perfectly uniform, the entropy is zero, and if the model is evenly split the entropy is 1. The split with the lowest entropy compared to the parent node and other splits is chosen. The lower the entropy, the better.

•	SVM: SVM is a supervised machine learning algorithm that can be used for both classification and regression tasks. SVM is mainly used in classification related problems. The algorithm plots each data item as a point in an n-dimensional space (where n is the number of features it possesses). where the value of each feature 
is the value of a specific coordinate. Classification is then performed 
by finding hyperplanes that distinguish the two classes very well. The support vector machine is the boundary that best separates the two classes (hyperplane/line). If your data requires nonlinear 
classification, SVM can use kernels. This is a function that takes a low-dimensional input space and transforms it into a high-dimensional space. H. They transform inseparable problems into separable problems.

•	Random Forest: A random forest is like a bootstrapping algorithm
using a decision tree model (CART). Random Forest attempts to build multiple CART models with different samples and different initial variables. For example, build a CART model using a random sample of 100 observations and 5 randomly chosen initial variables. Repeat this process (for example) 10 times and make a final prediction for 
each observation. The final prediction is a function of each and single prediction of the model. This final prediction is simply the average of each prediction. Random Forest provides more accurate predictions in many scenarios compared to simple CART/CHAID or regression models. These cases typically have a large number of predictors and a large sample size. This is so that we can obtain the variances of multiple input variables simultaneously,  allowing a large number of observations to participate in the prediction.


# PROBLEM STATEMENT: -
Spam classification has historically been viewed as a binary classification problem. This is where the most original aspects of our approach become apparent. We abandon the classification related problem in the favor of 
a regression problem which aims to predict the spam probability of a text message.

• Large datasets of spam SMS are not publicly available. Even so, we never expect training on these datasets to yield good performance in our context.
• Lack of a pipeline to transform SMS logs into a structured and clean   dataset.
• Apps are available in many countries and languages, adding to the 
  complexity.
• The ultimate model should be deployed and integrated into the  app's 
current infrastructure, taking the necessary precautions to avoid high costs and message delivery delays.
• Subjective Criteria for Labeling: Can religious propaganda be blocked 
even if it is not fraudulent or deceptive? What if the message is sent to  thousands of users?
• Message Ambiguity: Even humans have difficulty distinguishing 
between genuine messages and spam.

# CONCLUSION: - 

In this project, I have learned how to approach the problem and use data preprocessing and data visualization to draw useful conclusions from 
your data  that help you build better machine learning models.
To solve this classification problem, we used the Naive Bayes algorithm, specifically the Polynomial Naive Bayes algorithm. This is because it has 
the highest accuracy rating (i.e. the fewest false positives) and used his TFIDF for the vectorization method.

TF-IDF is an information retrieval technique that weights term frequency (TF) and its inverse document frequency (IDF). Each word or term that appears in the text has its own TF and IDF values.
Hyper parameter tuning of max_features further improved the model. 

The following techniques helped me understand how to build a text classification model and create a .pkl file to use over the network. This guide provides an overview of how to classify text messages as spam using different techniques.
You can probably imagine all the incredible possibilities and applications of this new information. This application may include chatbots, HR applications, and other systems. The possibilities are endless.

• Automatic labeling by frequency allows the meaningful creation of labeled 
  datasets.
• Spam detection can be refactored as a regression problem, and an added    message spam probability structure provides a more nuanced classification.
• The model successfully detects new spam patterns that were not detected      in the training data set.




















