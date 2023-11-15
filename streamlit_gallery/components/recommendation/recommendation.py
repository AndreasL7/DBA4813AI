import streamlit as st
import os
import google.generativeai as palm
import google.auth
import gc
import re
import numpy as np
import pandas as pd
from dotenv import load_dotenv

@st.cache_data
def load_data():
    url = "https://firebasestorage.googleapis.com/v0/b/studyszn.appspot.com/o/H2_physics_vectorised.csv?alt=media"
    df = pd.read_csv(url)
    return df

@st.cache_data
def apply_transformation(data):
    data['Embeddings'] = data['Embeddings'].apply(lambda x: np.array(eval(x)), 0) 
    return data

def display_open_ended_question(content, base_url="https://firebasestorage.googleapis.com/v0/b/studyszn.appspot.com/o/"):
    
    # Use a regular expression to find all placeholders like [image1], [image2], etc.
    placeholders = re.findall(r'\[image\d+\]', content)
    
    # Dynamically replace placeholders with image tags
    for placeholder in placeholders:
        # Extract the image number from the placeholder
        image_number = int(re.search(r'\d+', placeholder).group())
        # Create the image tag with the specified URL
        image_tag = f'![image]({base_url}image{image_number}.png?alt=media)'
        # Replace the placeholder with the image tag
        content = content.replace(placeholder, image_tag)
    # Create an HTML object and display it
    return content

def isOpenEndedQn(df, questionNo):
    return (pd.isnull(df.iloc[questionNo]['Option A Image']) and pd.isnull(df.iloc[questionNo]['Option A']))

def isImageMCQ(df, questionNo):
    return pd.notnull(df.iloc[questionNo]['Option A Image']) 

def isMCQWithoutOptions(df,questionNo):
    search_text = "<blockquote>\n<p>&nbsp;</p>\n</blockquote>\n"
    option_a_text = str(df.iloc[questionNo]['Option A'])
    return search_text in option_a_text

def Image_MCQ_replace_image_tags(row):
    text = row['Question']
    
    for Letter in ["A","B","C","D"]:
        text = text + "\n" + "[" + row[f'Option {Letter} Image'] + "] "
        
    for i in range(1, 7):
        try:
            if(pd.isnull(row[f'Qimage {i}'])):
                continue
            tag = f'[image{i}]'
            replacement = "[" + row[f'Qimage {i}'] + "] " #need a space right after!
            text = text.replace(tag, replacement)
    
        except KeyError:
            pass  # Handle the case where the Qimage column doesn't exist
    return text

def MCQ_text_Parser(row,text):
    pattern = r'<p>(.*?)<\/p>'
    
    for Letter in ["A","B","C","D"]:
        
        option = row[f'Option {Letter}']
        matches = re.findall(pattern, option, re.DOTALL)
        extracted_content = [match.strip() for match in matches]
        
        text = text + "\n" + "\n" + Letter + "\n" + ":     " + extracted_content[0] + "\n"
    return text


def replace_image_tags(row):
    text = row['Question']
    for i in range(1, 7):
        try:
            if(pd.isnull(row[f'Qimage {i}'])):
                continue
            tag = f'[image{i}]'
            replacement = "[" + row[f'Qimage {i}'] + "] " #need a space right after!
            text = text.replace(tag, replacement)
        except KeyError:
            pass  # Handle the case where the Qimage column doesn't exist
    return text

def parseQuestion(df, questionNum):
    if (isOpenEndedQn(df,questionNum)):
        content = replace_image_tags(df.iloc[questionNum])
        return (display_open_ended_question(content))
    else:
        row = df.iloc[questionNum]
        #is an mcq question
            
            
        if (isImageMCQ(df,questionNum)):
            return (display_open_ended_question(Image_MCQ_replace_image_tags(row)))

        elif (isMCQWithoutOptions(df,questionNum)):
            return display_open_ended_question(replace_image_tags(df.iloc[questionNum]))

        else:
            content = replace_image_tags(df.iloc[questionNum])
            return (display_open_ended_question(MCQ_text_Parser(df.iloc[questionNum],content)))
        
def replace_image_tags_answers(row):
    text = row['Answer Open']
    #text = text.replace("<p>", "").replace("</p>", "")
    for i in range(1, 7):
        try:
            if(pd.isnull(row[f'Answer Image{i}'])):
                continue
            tag = f'[image{i}]'
            replacement = "[" + row[f'Answer Image{i}'] + "] " #need a space right after!
            text = text.replace(tag, replacement)
        except KeyError:
            pass  # Handle the case where the Qimage column doesn't exist
    return text

def display_open_ended_answers(content, base_url="https://firebasestorage.googleapis.com/v0/b/studyszn.appspot.com/o/"):
    # Use a regular expression to find all placeholders like [image1], [image2], etc.
    placeholders = re.findall(r'\[image\d+\]', content)

    # Dynamically replace placeholders with HTML <img> tags
    for placeholder in placeholders:
        # Extract the image number from the placeholder
        image_number = int(re.search(r'\d+', placeholder).group())
        # Create the HTML <img> tag with the specified URL
        img_tag = f'<img src="{base_url}image{image_number}.png?alt=media" alt="image">'
        # Replace the placeholder with the HTML <img> tag
        content = content.replace(placeholder, img_tag)
    # Create an HTML object and display it
    return content

def parseAnswer(df, questionNum):
    if (isOpenEndedQn(df,questionNum)):
        content = replace_image_tags_answers(df.iloc[questionNum])
        return (display_open_ended_answers(content))
    elif not (isOpenEndedQn(df,questionNum)):
            row = df.iloc[questionNum]
            ans = "The answer to this MCQ question is: " + str(df.iloc[questionNum]['Answer Option'])
            return (ans)
    else:
            raise Exception("Error!")

def embed_fn(model, text):
    return palm.generate_embeddings(model=model, text=text)

def find_most_similar_questions(embedding, dataframe, top_n=5):
    # Stack embeddings into a 2D numpy array if they aren't already
    try:
        embeddings_stack = np.array(dataframe['Embeddings'].tolist())
        
        # Ensure the input embedding is a numpy array and has the correct shape (embedding_dimension,)
        input_embedding = np.array(embedding)
        
        # Check that the embedding dimensions align
        if embeddings_stack.shape[1] != input_embedding.shape[0]:
            raise ValueError(f"Embedding dimensions do not match: {embeddings_stack.shape[1]} in database vs {input_embedding.shape[0]} in input embedding.")
        
        dot_products = np.dot(embeddings_stack, input_embedding)
        
        top_indices = np.argpartition(dot_products, -top_n)[-top_n:]
        top_indices = top_indices[np.argsort(-dot_products[top_indices])]

        # Return the top N most similar question indices (adjusted for zero-based indexing)
        most_similar_indices = top_indices
        return list(most_similar_indices)
    
    except ValueError as e:
        return list()
        
def get_topic(questionnum, dataframe):
    return dataframe.iloc[questionnum]['topic']

def get_topics(questionnum, dataframe):
    return dataframe.iloc[questionnum]['related_topics'].strip('[]').replace("'", "")

@st.cache_data
def displayVideo():
    st.video("https://www.youtube.com/watch?v=chuKr5QUHoQ&list=PLrRPUj1fM--uRhxggaQv53XgcjwWmXu8-&index=3", start_time=0)

def main():
    
    gc.enable()
    load_dotenv()
    
    try:
        palm.configure(api_key = os.getenv('PALM_API_KEY'))
    except:
        os.environ['PALM_API_KEY'] = st.secrets['PALM_API_KEY']
        palm.configure(api_key = os.environ['PALM_API_KEY'])
    google.auth.default()
    
    # models = [m for m in palm.list_models() if 'embedText' in m.supported_generation_methods]
    # model = models[0]
    models = [m for m in palm.list_models() if 'embedText' in m.supported_generation_methods]

    model = models[0]

    # model = palm.get_model('models/chat-bison-001')
    
    html_with_css = """
    <style>
        body {
            font-family: Arial, sans-serif;
        }
        p {
            line-height: 1.5;
        }
    </style>
    """     

    # Initialization
    if ('result' not in st.session_state) or (st.session_state['result'] == None):
        st.warning("Please upload your question in the Introduction to get started!")
    else:
        st.info(st.session_state['result'])
        
    if "similarQs" not in st.session_state:
        st.session_state["similarQs"] = 0
    
    df = apply_transformation(load_data())
    
    if st.button("View Similar Questions"):
        if "predicted_topics" not in st.session_state:
            return None
            
        st.header("Similar Questions")
        similar_question_indices = find_most_similar_questions(np.array(embed_fn(model, st.session_state['result'])['embedding']), 
                                                               df.loc[lambda df_: df_['topic'] == st.session_state["predicted_topics"]],
                                                               top_n=5)
        if len(similar_question_indices) != 0:
            tab_labels = [f"Question {i+1}" for i in range(len(similar_question_indices))]
            tab1,tab2,tab3,tab4,tab5 = st.tabs(tab_labels)
            tabs = [tab1, tab2, tab3, tab4, tab5]

            for i,tab in enumerate(tabs):
                with tab:
                    question_expander = st.expander("Click to view question")
                    with question_expander:
                        question_output = html_with_css + parseQuestion(df.loc[lambda df_: df_['topic'] == st.session_state["predicted_topics"]], similar_question_indices[i])
                        st.markdown(question_output,unsafe_allow_html=True)

                    submitAnswer = st.expander("Click to view Answers")
                    with submitAnswer:
                        answer_output = html_with_css + parseAnswer(df.loc[lambda df_: df_['topic'] == st.session_state["predicted_topics"]], similar_question_indices[i])
                        st.markdown(answer_output, unsafe_allow_html=True)
                        displayVideo()
        else:
            st.error("Unfortunately, an error occurred. Please try another question.")
    
    gc.collect()

if __name__ == "__main__":
    main()