import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv("dataset.csv")
df = df[:31]

print(df['Job_Description'])


def f(user_input,location_input):
    # Sample user profile and job descriptions
    # user_profile = "Data scientist with expertise in machine learning"
    user_profile = user_input
    job_descriptions = np.array(df['Job_Description'])
    print(job_descriptions)

    # Combine user profile and job descriptions for vectorization

    all_text = [user_profile] + job_descriptions

    # Create TF-IDF vectorizer and fit on combined text data
    vectorizer = TfidfVectorizer()
    text_vectors = vectorizer.fit_transform(all_text)

    # Calculate cosine similarity between user profile vector and job description vectors
    user_vector = text_vectors[0]  # User profile vector
    job_vectors = text_vectors[1:]  # Job description vectors

    cosine_similarities = cosine_similarity(user_vector, job_vectors)[0]

    # Rank job descriptions based on cosine similarities
    sorted_indices = np.argsort(cosine_similarities)[::-1]  # Sort in descending order

    # Recommend top N jobs
    num_recommendations = 3
    recommended_jobs = [job_descriptions[idx] for idx in sorted_indices[:num_recommendations]]

    print("Recommended Jobs:")
    output = "\n"
    output2 = '\n'
    index = []
    for i, job in enumerate(recommended_jobs, 1):
        output += str(i)+". " +job+"\n"
        index.append(int(np.where(job_descriptions == job)[0][0]))
    j = 1
    print(index)
    location = np.array(df['Location'])
    print(location[0])
    print(df['Location'].iloc[[0]])
    for i in index:
        print(str(location[i]) ,"  &&&&  " ,location_input)
        if (str(location[i]) == location_input):
            print("===============================================")
            output2 += str(j)+". " +job_descriptions[i]+"\n"
            j+=1
    print(output2)

    return output,output2

# Title
st.title("Job Recommondation System")

# Input
user_input = st.text_input("Enter User Profile", "")
location_input = st.text_input("Enter Location", "")
# Button
button_clicked = st.button("Submit")

# Output
if button_clicked:
    output,output2 = f(user_input,location_input)
    st.write("Recommended jobs :", output)
    st.write("Jobs available based on location :",output2)


