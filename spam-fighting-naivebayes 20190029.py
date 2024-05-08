import os
import email
# In[2]:
DATA_DIR = 'datasets/trec07p/data/'
LABELS_FILE = 'datasets/trec07p/full/index'
#%%
def flatten_to_string(parts):
    ret = []
    if type(parts) == str:
        ret.append(parts)
    elif type(parts) == list:
        for part in parts:
            ret += flatten_to_string(part)
    elif parts.get_content_type == 'text/plain':
        ret += parts.get_payload()
    return ret
#%%
# Extract subject and body text from a single email file
def extract_email_text(path): # I Will be modifying here add the source from + the subject and body
    # Load a single email from an input file
    with open(path,"r", errors='ignore') as f:
        msg = email.message_from_file(f)
    if not msg:
        return ""
    # Read the email subject
    subject = msg['Subject']
    if not subject:  subject = ""

    # Read the email body
    body = ' '.join(m for m in flatten_to_string(msg.get_payload()) if type(m) == str)
    if not body:  body = ""
    #extract from to
    from_header = msg['From']
    if from_header:
        domain = from_header.split('@')[-1] if '@' in from_header else ""
    else:
        domain = ""
    #i will try the accuracy if i extract the dates, maybe some emails are sent at specific times that could be useful
    recv_header=msg['Received']
    date=recv_header.split(';')[-1] if ';' in recv_header else ""    
    return subject + ' ' + body + ' ' + domain + ' ' + date

#%%
def read_email_files():
    X = []
    y = [] 
    for i in range(len(labels)):
        filename = 'inmail.' + str(i+1)
        # print(filename,"=====")
        email_str = extract_email_text(os.path.join(DATA_DIR, filename))
        X.append(email_str)
        y.append(labels[filename])
    return X, y
#%%
labels = {}
# Read the labels
with open(LABELS_FILE) as f:
    for line in f:
        line = line.strip()
        label, key = line.split()
        labels[key.split('/')[-1]] = 1 if label.lower() == 'ham' else 0

X, y = read_email_files()

#%%
from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = \
     train_test_split(X, y, train_size=0.7, random_state=2)

# #%%
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
X_train_vector = vectorizer.fit_transform(X_train)
X_test_vector = vectorizer.transform(X_test)
# #%%
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score,recall_score,f1_score
#Initialize the classifier and make label predictions
mnb = MultinomialNB()
mnb.fit(X_train_vector, y_train)
y_pred = mnb.predict(X_test_vector)

# # Print results
print(classification_report(y_test, y_pred, target_names=['Spam', 'Ham']))
print('Classification accuracy {:.1%}'.format(accuracy_score(y_test, y_pred)))
print(precision_score(y_test, y_pred))
print(recall_score(y_test, y_pred))
print(f1_score(y_test, y_pred))

print(confusion_matrix(y_test, y_pred))

DATA_DIR2 = 'datasets2/trec07p/data/'
def read_unlabled():
    X2 = []
    email_files = ['inmail.75415', 'inmail.75416', 'inmail.75417', 'inmail.75418', 'inmail.75419']
    for filename in email_files:
        email_str = (os.path.join(DATA_DIR2, filename))
        prediction= unlabled_prediciton(email_str)
        X2.append(prediction)
    return X2

def unlabled_prediciton(path2):
    email_text= extract_email_text(path2)
    x2_test=vectorizer.transform([email_text])
    pred=mnb.predict(x2_test)
    if pred[0] ==0:
        return 'Spam'
    else:
        return 'Ham'



X2 = read_unlabled()
for prediction in X2:
    print(f'Email is {prediction}')
