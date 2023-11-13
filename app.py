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