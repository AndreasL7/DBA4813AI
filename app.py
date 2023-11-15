import streamlit as st
from streamlit_lottie import st_lottie
import requests

from streamlit_gallery import components
from streamlit_gallery.utils.page import page_group

@st.cache_data
def load_lottie_url(url: str):
    
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

def main():

    # Custom CSS
    styles = """
        <style>
            body {
                background-color: #FAF3E0; 
                font-family: "Arial", sans-serif;
            }
            
            h1 {
                font-family: "Georgia", monospace; 
                color: #3E2723;
            }
            
            .stButton>button {
                background-color: #575735;
                color: white !important;
            }
        </style>
    """
    
    st.markdown(styles, unsafe_allow_html=True)
    
    #Lottie
    lottie_url = "https://lottie.host/fc73c427-341f-42b1-9e43-28d52e9577f4/jsprOQ9sKu.json"  # Sample URL, replace with your desired animation
    lottie_animation = load_lottie_url(lottie_url)
    st_lottie(lottie_animation, speed=1, width=200, height=200)
    
    st.title('DBA4813')
    st.markdown("""Due to resource constraint provided by Streamlit Sharing, only permitted users are allowed access. Please note that the app interface is not flawless, occasional state rollback may occur. Nevertheless, the app serves its purpose of demonstrating the model's performance.""")    

    st.subheader("**Welcome to Your Interactive Physics Practice Assistant!**")
    st.markdown("""
    ### 
    🚀 **Embark on a Journey of Discovery and Mastery in Physics!**

    Unlock the mysteries of physics and transform the way you learn with our innovative practice tool. Designed specifically for Singapore Junior College students, this platform is your go-to resource for mastering physics.

    **Here’s how our intuitive tool works:**

    1. **Capture Your Challenge:** Got a physics question? Snap and upload a screenshot of it.
    2. **Magic of Technology:** Our system reads and extracts the text from your question with precision.
    3. **AI-Powered Prediction:** Our advanced tech analyses the question and predicts its specific subtopic. It's like having your own AI tutor!
    4. **Tailored Practice:** We then match your question with similar ones from our extensive database, based on the identified subtopic. This personalised approach ensures you practice what's most relevant to you.
    5. **Master and Conquer:** With a selection of targeted practice questions, hone your skills and gain confidence in tackling a wide range of physics problems.

    **Explore a Universe of Physics Subtopics:**

    Our tool is equipped to assist you across a diverse range of physics areas. Here are the subtopics you can explore:

    - Electric Fields
    - Wave Motion
    - Temperature and Ideal Gasses
    - Nuclear Physics
    - Forces
    - D.C. Circuits
    - Gravitational Field
    - Quantum Physics

    Whether you're unraveling the complexities of Nuclear Physics or exploring the intricacies of D.C. Circuits, our tool is designed to cater to your learning needs.

    **Begin Your Physics Adventure Today!**

    Step into a world where physics is not just a subject, but an exciting journey of exploration and understanding. Upload your first question now and see where physics can take you!            
    """)
    page = page_group("p")

    with st.sidebar:
        st.title("🕵️‍♂️ Tutor's Gallery")
        st.caption("where Education meets AI")
        st.write("")
        st.markdown('Made by <a href="https://www.linkedin.com/in/andreaslukita7/">Andreas Lukita</a>', unsafe_allow_html=True)

        with st.expander("⏳ COMPONENTS", True):
            page.item("Introduction", components.show_introduction, default=True)
            page.item("Recommendation⭐", components.show_recommendation)

    page.show()

if __name__ == "__main__":
    main()