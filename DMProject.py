import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
from scipy.stats import randint

# prep
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import preprocessing
from sklearn.datasets import make_classification
from sklearn.preprocessing import binarize, LabelEncoder, MinMaxScaler

# models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

# Validation libraries
from sklearn import metrics
from sklearn.metrics import accuracy_score, mean_squared_error, precision_recall_curve
from sklearn.model_selection import cross_val_score

#Neural Network
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import RandomizedSearchCV

#Bagging
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier

#Naive bayes
from sklearn.naive_bayes import GaussianNB

#Stacking
from mlxtend.classifier import StackingClassifier

# Any results you write to the current directory are saved as output.

# reading in CSV's from a file path
train_df = pd.read_csv('OSMI Mental Health in Tech Survey 2017.csv')

# # Pandas: whats the data row count?
print(train_df.shape)

# # Pandas: whats the distribution of the data?
print(train_df.describe())

# # Pandas: What types of data do i have?
print(train_df.info())




#dealing with missing data
#Get rid of the unnecessary variables
train_df = train_df.drop(['#'], axis= 1)
train_df = train_df.drop(['Describe the conversation with coworkers you had about your mental health including their reactions.'], axis= 1)
train_df = train_df.drop(['Would you have felt more comfortable talking to your previous employer about your physical health or your mental health?'], axis= 1)
train_df = train_df.drop(['Describe the conversation you had with your employer about your mental health, including their reactions and what actions were taken to address your mental health issue/questions.'], axis= 1)
train_df = train_df.drop(['Would you have been willing to discuss your mental health with your direct supervisor(s)?'], axis= 1)
train_df = train_df.drop(['Describe the conversation your coworker had with you about their mental health (please do not use names)..1'], axis= 1)
train_df = train_df.drop(['<strong>Would you have been willing to discuss your mental health with your coworkers at previous employers?</strong>'], axis= 1)
train_df = train_df.drop(['Describe the conversation you had with your previous employer about your mental health, including their reactions and actions taken to address your mental health issue/questions.'], axis= 1)
train_df = train_df.drop(['Describe the conversation you had with your previous coworkers about your mental health including their reactions.'], axis= 1)
train_df = train_df.drop(['Describe the conversation your coworker had with you about their mental health (please do not use names).'], axis= 1)
train_df = train_df.drop(['How has it affected your career?'], axis= 1)
train_df = train_df.drop(['Briefly describe what you think the industry as a whole and/or employers could do to improve mental health support for employees.'], axis= 1)
train_df = train_df.drop(['Describe the circumstances of the supportive or well handled response.'], axis= 1)
train_df = train_df.drop(['Describe the circumstances of the badly handled or unsupportive response.'], axis= 1)
train_df = train_df.drop(['Network ID'], axis= 1)
train_df = train_df.drop(['Start Date (UTC)'], axis= 1)
train_df = train_df.drop(['Submit Date (UTC)'], axis= 1)
train_df = train_df.drop(['Other'], axis= 1)
train_df = train_df.drop(['Other.1'], axis= 1)
train_df = train_df.drop(['Why or why not?'], axis= 1)
train_df = train_df.drop(['Why or why not?.1'], axis= 1)



train_df.isnull().sum().max() #just checking that there's no missing data missing...
print(train_df.columns)
print(train_df.head(5))



# Assign default values for each data type
defaultInt = 0
defaultString = 'N/A'
defaultFloat = 0.0
# Create lists by data tpe
intFeatures = ['self_employed', 'Is your employer primarily a tech company/organization?',
               'Is your primary role within your company related to tech/IT?',
               'Have you ever discussed your mental health with your employer?',
               'Have you ever discussed your mental health with coworkers?',
               'Have you ever had a coworker discuss their or another coworker\'s mental health with you?',
               'Overall, how much importance does your employer place on physical health?',
               'Overall, how much importance does your employer place on mental health?',
               'Do you have medical coverage (private insurance or state-provided) that includes treatment of mental health disorders?',
               '<strong>Do you have previous employers?</strong>',
               'Was your employer primarily a tech company/organization?',
               'Did you ever discuss your mental health with your previous employer?',
               'Did you ever discuss your mental health with a previous coworker(s)?',
               'Did you ever have a previous coworker discuss their or another coworker\'s mental health with you?',
               'Overall, how much importance did your previous employer place on physical health?',
               'Overall, how much importance did your previous employer place on mental health?',
               'Have you ever sought treatment for a mental health disorder from a mental health professional?',
               'How willing would you be to share with friends and family that you have a mental illness?',
               'Are you openly identified at work as a person with a mental health issue?',
               'Has being identified as a person with a mental health issue affected your career?',
               'If they knew you suffered from a mental health disorder, how do you think that team members/co-workers would react?',
               'Overall, how well do you think the tech industry supports employees with mental health issues?',
               'Would you be willing to talk to one of us more extensively about your experiences with mental health issues in the tech industry? (Note that all interview responses would be used <em>anonymously</em> and only with your permission.)',
               'Age']


stringFeatures = ['How many employees does your company or organization have?',
                  'Does your employer provide mental health benefits as part of healthcare coverage?',
                  'Do you know the options for mental health care available under your employer-provided health coverage?',
                  'Has your employer ever formally discussed mental health (for example, as part of a wellness campaign or other official communication)?',
                  'Does your employer offer resources to learn more about mental health disorders and options for seeking help?',
                  'Is your anonymity protected if you choose to take advantage of mental health or substance abuse treatment resources provided by your employer?',
                 'If a mental health issue prompted you to request a medical leave from work, how easy or difficult would it be to ask for that leave?',
                  'Would you feel more comfortable talking to your coworkers about your physical health or your mental health?',
                  'Would you feel comfortable discussing a mental health issue with your direct supervisor(s)?',
                  'Describe the conversation you had with your employer about your mental health, including their reactions and what actions were taken to address your mental health issue/questions.',
                  'Would you feel comfortable discussing a mental health issue with your coworkers?',
                  'Do you know local or online resources to seek help for a mental health issue?',
                 '<strong>If you have been diagnosed or treated for a mental health disorder, do you ever reveal this to clients or business contacts?</strong>',
                  'If you have revealed a mental health disorder to a client or business contact, how has this affected you or the relationship?',
                  '<strong>If you have been diagnosed or treated for a mental health disorder, do you ever reveal this to coworkers or employees?</strong>',
                  'If you have revealed a mental health disorder to a coworker or employee, how has this impacted you or the relationship?',
                  'Do you believe your productivity is ever affected by a mental health issue?',
                  'If yes, what percentage of your work time (time performing primary or secondary job functions) is affected by a mental health issue?',
                  '<strong>Have your previous employers provided mental health benefits?</strong>',
                  '<strong>Were you aware of the options for mental health care provided by your previous employers?</strong>',
                  'Did your previous employers ever formally discuss mental health (as part of a wellness campaign or other official communication)?',
                  'Did your previous employers provide resources to learn more about mental health disorders and how to seek help?',
                  'Was your anonymity protected if you chose to take advantage of mental health or substance abuse treatment resources with previous employers?',
                  'Do you currently have a mental health disorder?',
                  'Have you ever been diagnosed with a mental health disorder?',
                  'Anxiety Disorder (Generalized, Social, Phobia, etc)',
                  'Mood Disorder (Depression, Bipolar Disorder, etc)',
                  'Psychotic Disorder (Schizophrenia, Schizoaffective, etc)',
                  'Eating Disorder (Anorexia, Bulimia, etc)',
                  'Attention Deficit Hyperactivity Disorder',
                  'Personality Disorder (Borderline, Antisocial, Paranoid, etc)',
                  'Obsessive-Compulsive Disorder',
                  'Post-traumatic Stress Disorder',
                  'Stress Response Syndromes',
                  'Dissociative Disorder',
                  'Substance Use Disorder',
                  'Addictive Disorder',
                  'Have you had a mental health disorder in the past?',
                  'Do you have a family history of mental illness?',
                  'If you have a mental health disorder, how often do you feel that it interferes with your work <strong>when being treated effectively?</strong>',
                  'If you have a mental health disorder, how often do you feel that it interferes with your work <strong>when <em>NOT</em> being treated effectively (i.e., when you are experiencing symptoms)?</strong>',
                  'Have your observations of how another individual who discussed a mental health issue made you less likely to reveal a mental health issue yourself in your current workplace?',
                  'Would you be willing to bring up a physical health issue with a potential employer in an interview?',
                  'Would you bring up your mental health with a potential employer in an interview?',
                  '<strong>Have you observed or experienced an unsupportive or badly handled response to a mental health issue in your current or previous workplace?</strong>',
                  '<strong>Have you observed or experienced supportive or well handled response to a mental health issue in your current or previous workplace?</strong>',
                  'If there is anything else you would like to tell us that has not been covered by the survey questions, please use this space to do so.',
                  'Gender',
                  'What country do you <strong>live</strong> in?',
                  'What US state or territory do you <strong>live</strong> in?',
                  'What is your race?',
                  'What country do you <strong>work</strong> in?',
                  'What US state or territory do you <strong>work</strong> in?']
floatFeatures = []

for feature in train_df:
    if feature in intFeatures:
        train_df[feature] = train_df[feature].fillna(defaultInt)
    elif feature in stringFeatures:
        train_df[feature] = train_df[feature].fillna(defaultString)
    elif feature in floatFeatures:
        train_df[feature] = train_df[feature].fillna(defaultFloat)
    else:
        print('Error: Feature %s not recognized.' % feature)
print(train_df.head(5))



#clean 'Gender'
#Slower case all columm's elements
gender = train_df['Gender'].str.lower()

#Select unique elements
gender = train_df['Gender'].unique()

#Made gender groups
male_str = ["male", "m", "male-ish", "man", "mal", "male (cis)", "make", "male ", "man", "msle", "mail", "malr","cis man", "cis male", "male/androgynous ", "cis hetero male",
            "male (hey this is the tech industry you're talking about)", "God King of the Valajar", "Cis male", "Cis-male", "male-ish", "male, cis", "dude", "cis-male", "Cis-male", "cis male "]
trans_str = ["trans-female", "something kinda male?", "queer/she/they", "non-binary","nah", "all", "enby", "fluid", "genderqueer", "genderfluid", "uhhhhhhhhh fem genderqueer?", "n/a",
             "nonbinary", "Transfeminine", "trans woman", "\-", "non binary", "contextual", "sometimes", "agender/genderfluid", "none", "Genderqueer demigirl", "Genderqueer/non-binary", "Transfeminine"]
female_str = ["cis female", "f", "female", "woman",  "femake", "female ","cis-female/femme", "female (cis)", "femail", "femalw", "my sex is female.", "female-ish", "f, cisgender",
              "cis-female", "woman-identified", "female (cis)", "female (cisgender)", "Female (cis) "]

for (row, col) in train_df.iterrows():
    if str.lower(col.Gender) in male_str:
        train_df['Gender'].replace(to_replace=col.Gender, value='male', inplace=True)
    if str.lower(col.Gender) in female_str:
        train_df['Gender'].replace(to_replace=col.Gender, value='female', inplace=True)
    if str.lower(col.Gender) in trans_str:
        train_df['Gender'].replace(to_replace=col.Gender, value='trans', inplace=True)

stk_list = ['A little about you', 'p']
train_df = train_df[~train_df['Gender'].isin(stk_list)]
print(train_df['Gender'].unique())


#complete missing age with mean
train_df['Age'].fillna(train_df['Age'].median(), inplace = True)
# Fill with media() values < 18 and > 120
s = pd.Series(train_df['Age'])
s[s<18] = train_df['Age'].median()
train_df['Age'] = s
s = pd.Series(train_df['Age'])
s[s>120] = train_df['Age'].median()
train_df['Age'] = s
#Ranges of Age
train_df['age_range'] = pd.cut(train_df['Age'], [0,20,30,65,100], labels=["0-20", "21-30", "31-65", "66-100"], include_lowest=True)

# Encoding data
labelDict = {}
for feature in train_df:
    le = preprocessing.LabelEncoder()
    le.fit(train_df[feature])
    le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    train_df[feature] = le.transform(train_df[feature])
    # Get labels
    labelKey = 'label_' + feature
    labelValue = [*le_name_mapping]
    labelDict[labelKey] = labelValue

for key, value in labelDict.items():
    print(key, value)
print(train_df.head())

# missing data
total = train_df.isnull().sum().sort_values(ascending=False)
percent = (train_df.isnull().sum() / train_df.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)
print(missing_data)



#correlation matrix
corrmat = train_df.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True);
plt.show()


# Distribution and density by Age
plt.figure(figsize=(12,8))
sns.distplot(train_df["Age"], bins=24)
plt.title("Distribuition and density by Age")
plt.xlabel("Age")
plt.show()


# Relation between age and mental health disorder
g = sns.FacetGrid(train_df, col='Have you ever been diagnosed with a mental health disorder?', size=5)
g = g.map(sns.distplot, "Age")
plt.show()


#Relation between gender and mental health disorder
g = sns.FacetGrid(train_df, col='Have you ever been diagnosed with a mental health disorder?', size=5)
g = g.map(sns.distplot, 'Gender')
plt.show()

plt.figure(figsize=(12,8))
labels = labelDict['label_Gender']
g = sns.countplot(x="Have you ever been diagnosed with a mental health disorder?", data=train_df)
g.set_xticklabels(labels)
plt.title('Mental Illness Distribution')
plt.show()


# Relation between age and mental health disorder
o = labelDict['label_age_range']
g = sns.factorplot(x="age_range", y="Have you ever been diagnosed with a mental health disorder?", hue="Gender", data=train_df, kind="bar",  ci=None, size=5, aspect=2, legend_out = True)
g.set_xticklabels(o)
plt.title('Probability of mental health condition')
plt.ylabel('Probability x 50')
plt.xlabel('Age')
# replace legend labels
new_labels = labelDict['label_Gender']
for t, l in zip(g._legend.texts, new_labels): t.set_text(l)
g.fig.subplots_adjust(top=0.9,right=0.8)
plt.show()


o = labelDict['label_<strong>Have your previous employers provided mental health benefits?</strong>']
g = sns.factorplot(x="<strong>Have your previous employers provided mental health benefits?</strong>", y="Have you ever been diagnosed with a mental health disorder?",
                   hue="Gender", data=train_df, kind="bar", ci=None, size=5, aspect=2, legend_out = True)
g.set_xticklabels(o)
plt.title('Probability of mental health condition')
plt.ylabel('Probability x 50')
plt.xlabel('Mental Health Benefits')
# replace legend labels
new_labels = labelDict['label_Gender']
for t, l in zip(g._legend.texts, new_labels): t.set_text(l)
g.fig.subplots_adjust(top=0.9,right=0.8)
plt.show()



# Scaling Age
scaler = MinMaxScaler()
train_df['Age'] = scaler.fit_transform(train_df[['Age']])
pd.set_option('display.max_columns', None)
print(train_df.head())



# define X and y
feature_cols = ['Age', 'Gender', 'Do you have a family history of mental illness?', 'Is your employer primarily a tech company/organization?',
                '<strong>Have your previous employers provided mental health benefits?</strong>',
                'Is your primary role within your company related to tech/IT?',
                'Overall, how much importance does your employer place on mental health?',
                'If yes, what percentage of your work time (time performing primary or secondary job functions) is affected by a mental health issue?',
                'Did your previous employers provide resources to learn more about mental health disorders and how to seek help?']
X = train_df[feature_cols]
y = train_df['Have you ever been diagnosed with a mental health disorder?']

# split X and y into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)

# Create dictionaries for final graph
# Use: methodDict['Stacking'] = accuracy_score
methodDict = {}
rmseDict = ()



# Build a forest and compute the feature importances
forest = ExtraTreesClassifier(n_estimators=250,
                              random_state=0)
forest.fit(X, y)
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

labels = []
for f in range(X.shape[1]):
    labels.append(feature_cols[f])

# Plot the feature importances of the forest
plt.figure(figsize=(12, 8))
plt.title("Feature importances")
plt.bar(range(X.shape[1]), importances[indices],
        color="r", yerr=std[indices], align="center")
plt.xticks(range(X.shape[1]), labels, rotation='vertical')
plt.xlim([-1, X.shape[1]])
plt.show()