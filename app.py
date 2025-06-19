from PIL import Image
import streamlit as st
from streamlit_lottie import st_lottie
import requests

# --- GENERAL SETTINGS ---
st.set_page_config(page_title="Digital Resume | Narasimha Naidu Nagam", page_icon=":waving_hand:", layout="wide")

# --- PERSONAL INFO ---
NAME = "Narasimha Naidu Nagam"
DESCRIPTION = "AI/ML Trainee | Skilled in Data Processing, Model Development, and Deployment using Streamlit, Flask, Docker & Kubernetes"
EMAIL = "nagamnaidu3@gmail.com"
PHONE = "+91 9492252452, +91 9515874452"
LINKEDIN = "https://www.linkedin.com/in/nagam-narasimha-naidu-836a92176/"
GITHUB = "https://github.com/NagamNaidu"
RESUME_LINK = "https://github.com/NagamNaidu/My-Resume/blob/main/Narasimha_Naidu_Nagam_Resume.pdf"

# --- UTILITY FUNCTION TO LOAD ANIMATION ---
import urllib.parse

def load_lottieurl(url):
    try:
        # Validate URL
        result = urllib.parse.urlparse(url)
        if not all([result.scheme, result.netloc]):
            st.error(f"⚠️ Invalid URL: {url}")
            return None

        r = requests.get(url)
        if r.status_code != 200:
            st.error(f"⚠️ Failed to load animation from {url}. Status code: {r.status_code}")
            return None
        return r.json()
    except Exception as e:
        st.error(f"⚠️ Error loading animation from {url}: {e}")
        return None

tech_images = {
    "Python": "https://upload.wikimedia.org/wikipedia/commons/c/c3/Python-logo-notext.svg",
    "JavaScript": "https://upload.wikimedia.org/wikipedia/commons/6/6a/JavaScript-logo.png",
    "Streamlit": "https://streamlit.io/images/brand/streamlit-mark-color.png",
    "ReactJS": "https://upload.wikimedia.org/wikipedia/commons/a/a7/React-icon.svg",
    "Docker": "https://www.docker.com/wp-content/uploads/2022/03/Moby-logo.png",
    "Kubernetes": "kubernetes.png",
    "NumPy": "numpy.png",
    "Pandas": "https://upload.wikimedia.org/wikipedia/commons/e/ed/Pandas_logo.svg",
    "Matplotlib": "matplotlib.png",
    "Scikit-learn": "https://upload.wikimedia.org/wikipedia/commons/0/05/Scikit_learn_logo_small.svg",
    "Seaborn": "https://seaborn.pydata.org/_static/logo-wide-lightbg.svg",
    "TensorFlow": "tensorflow.png"
}
# --- CSS STYLING ---
st.markdown("""
    <style>
    .main-card {
        background: #ffffff;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.1);
        margin-bottom: 2rem;
    }
    .skill-bar {
        background-color: #eee;
        border-radius: 10px;
        padding: 5px;
        margin-bottom: 10px;
    }
    .skill-level {
        background-color: #4CAF50;
        height: 20px;
        border-radius: 10px;
        text-align: right;
        padding-right: 10px;
        color: white;
        font-size: 12px;
    }
    .project-card {
        background: #f9f9f9;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# --- HEADER ---
# --- HEADER ---
with st.container():
    col1, col2 = st.columns([1, 3])
    with col1:
        try:
            st.image("1000061147.jpg", width=150, caption="Profile Picture")
        except FileNotFoundError:
            st.error("Profile image not found. Please make sure '1000061147.jpg' is in the same folder as your app.py")
    with col2:
        st.markdown(f"""
        <div class="main-card">
            <h1 style='text-align: center;'>{NAME}</h1>
            <p style='text-align: center;'>{DESCRIPTION}</p>
            <p style='text-align: center;'>📧 {EMAIL} | 📞 {PHONE}</p>
            <p style='text-align: center;'>
                <a href='{LINKEDIN}' target='_blank'>LinkedIn</a> |
                <a href='{GITHUB}' target='_blank'>GitHub</a> |
                <a href='{RESUME_LINK}' download>Download Resume</a>
            </p>
        </div>
        """, unsafe_allow_html=True)

# --- TABS ---
tabs = st.tabs(["About Me", "Career Objective", "Experience", "Technologies", "Projects", "Education", "Contact Me"])

def show_image(image_path, height=200):
    try:
        image = Image.open(image_path)
        st.image(image, width=height)
    except FileNotFoundError:
        st.error(f"Image not found: {image_path}")
    except Exception as e:
        st.error(f"Error loading image: {e}")
        st.write(e)

# --- ABOUT ME ---
with tabs[0]:
    st.subheader("About Me")
    st.image("aboutme.jpeg", width=200)
    st.write("I am an enthusiastic AI/ML trainee with a strong foundation in model development, deployment, and automation workflows. Passionate about learning and exploring innovative ML applications.")

# --- CAREER OBJECTIVE ---
with tabs[1]:
    st.subheader("Career Objective")
    st.image("career objective.jpeg", width=200)
    st.info("""
    Highly motivated and quick-learning individual seeking a machine learning trainee position to leverage foundational knowledge and contribute to innovative projects.
    Passionate about applying ML to solve real-world problems.
    """)

# --- EXPERIENCE ---
with tabs[2]:
    st.subheader("Experience")
    st.image("experience.jpeg", width=200)
    st.markdown("""
    **AI/ML Trainee – Lyros Technology Pvt. Ltd | Jan 2025 – Present**

    **A. Data Collection & Processing:**
    - Assist in collecting, cleaning, and structuring datasets for AI/ML training.
    - Ensure data integrity and optimize data workflows.

    **B. ML Model Development & Testing:**
    - Support in training, evaluating models; suggest improvements for accuracy.

    **C. AI Application Optimization:**
    - Analyze AI apps, propose optimizations to improve scalability.

    **D. Research & Experimentation:**
    - Contribute to research; integrate new ML techniques.

    **E. Automation & Workflow Enhancement:**
    - Write scripts to automate tasks; integrate with DevOps workflows.

    **F. Documentation & Reporting:**
    - Maintain clear records of ML experiments and findings.

    **G. Collaboration & Team Support:**
    - Work closely with cross-functional teams; contribute to brainstorming.
    """)

# --- TECHNOLOGIES ---
with tabs[3]:
    st.subheader("Technologies")
    st.image("technologies and tools.jpeg", width=200)
    tech_cols = st.columns(2)

    with tech_cols[0]:
        st.markdown("### Programming & Tools")
        skills = {
            "Python": 90,
            "JavaScript": 80,
            "Streamlit": 85,
            "ReactJS": 70,
            "Docker": 75,
            "Kubernetes": 65
        }
        for skill, percent in skills.items():
            col1, col2 = st.columns([1, 2])
            with col1:
                try:
                    st.image(tech_images[skill], width=100)
                except:
                    st.warning(f"⚠️ Could not load image for {skill}")
            with col2:
                st.markdown(f"{skill}")
                st.markdown(f"""
                <div class='skill-bar'>
                    <div class='skill-level' style='width: {percent}%;'>{percent}%</div>
                </div>
                 """, unsafe_allow_html=True)

    with tech_cols[1]:
        st.markdown("### Libraries & Frameworks")
        libs = {
            "NumPy": 85,
            "Pandas": 90,
            "Matplotlib": 80,
            "Scikit-learn": 85,
            "Seaborn": 75,
            "TensorFlow": 60
        }
        for lib, percent in libs.items():
            col1, col2 = st.columns([1, 2])
            with col1:
                try:
                    st.image(tech_images[lib], width=100)
                except:
                    st.warning(f"⚠️ Could not load image for {lib}")
            with col2:
                st.markdown(f"{lib}")
                st.markdown(f"""
                <div class='skill-bar'>
                    <div class='skill-level' style='width: {percent}%;'>{percent}%</div>
                </div>
                """, unsafe_allow_html=True)

# --- PROJECTS ---
with tabs[4]:
    st.subheader("Projects")
    st.image("projects.jpeg", width=200)

    projects = {
        "Swiggy Clone": {
            "description": "Developed using ReactJS and TailwindCSS. Used Redux Toolkit and Hooks for managing state.",
            "view_link": None, # Replace with actual link if available
        },
        "Disease Prediction (ML) Demo": {
            "description": "Interactive demo for disease prediction using a Logistic Regression model.",
            "view_link": "#disease-prediction-demo"
        },
        "Patient Dataset ML Preprocessing Demo": {
            "description": "Interactive demo for patient dataset preprocessing using StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder, LabelEncoder and SimpleImputer.",
            "view_link": "#patient-dataset-preprocessing-demo"
        },
        "Tax Calculator App": {
            "description": "Created using Streamlit to calculate Indian income tax slabs. Deployed and shared via Streamlit Cloud.",
            "view_link": None, # Replace with actual link if available
        },
        "Iris Dataset with Flask & Random Forest": {
            "description": "Built a prediction app using petal/sepal data. Used Flask backend and deployed for interactive use.",
            "view_link": None, # Replace with actual link if available
        }
    }

    for project_name, project_details in projects.items():
        with st.container():
            st.markdown(f"<div class='project-card'>", unsafe_allow_html=True)
            st.subheader(project_name)
            st.write(project_details["description"])

            if project_name == "Disease Prediction (ML) Demo":
                st.subheader("Disease Prediction Demo")
                dataset_options = ["Heart Disease", "Diabetes", "Cancer"]
                selected_dataset = st.selectbox("Select Dataset", dataset_options)

                # Add input fields for symptoms based on the selected dataset
                if selected_dataset == "Heart Disease":
                    age = st.number_input("Age", min_value=0, max_value=120, value=50)
                    cholesterol = st.number_input("Cholesterol", min_value=0, max_value=500, value=200)
                    # Add more input fields for other symptoms

                    import pandas as pd
                    from sklearn.model_selection import train_test_split
                    from sklearn.linear_model import LogisticRegression
                    from sklearn.metrics import accuracy_score

                    # Load the dataset
                    data = {'age': [50, 60, 70, 40, 30], 'cholesterol': [200, 250, 300, 180, 150], 'target': [1, 0, 1, 0, 0]}
                    df = pd.DataFrame(data)

                    # Prepare the data
                    X = df[['age', 'cholesterol']]
                    y = df['target']

                    # Split the data into training and testing sets
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                    # Train the model
                    model = LogisticRegression()
                    model.fit(X_train, y_train)

                    if st.button("Predict"):
                        # Make predictions
                        new_data = pd.DataFrame({'age': [age], 'cholesterol': [cholesterol]})
                        prediction = model.predict(new_data)[0]

                        # Display the prediction
                        if prediction == 1:
                            st.write("Prediction: High risk of heart disease")
                        else:
                            st.write("Prediction: Low risk of heart disease")

                        # Evaluate the model
                        y_pred = model.predict(X_test)
                        accuracy = accuracy_score(y_test, y_pred)
                        st.write(f"Accuracy: {accuracy}")

                    # Add code to visualize the prediction

            elif project_name == "Patient Dataset ML Preprocessing Demo":
                st.subheader("Patient Dataset Preprocessing Demo")
                uploaded_file = st.file_uploader("Upload Patient Dataset (CSV)", type=["csv"])

                if uploaded_file is not None:
                    import pandas as pd
                    df = pd.read_csv(uploaded_file)
                    st.dataframe(df)

                    scaling_options = ["None", "StandardScaler", "MinMaxScaler", "RobustScaler"]
                    selected_scaling = st.selectbox("Scaling Technique", scaling_options)

                    encoding_options = ["None", "OneHotEncoder", "LabelEncoder"]
                    selected_encoding = st.selectbox("Encoding Technique", encoding_options)

                    imputation_options = ["None", "SimpleImputer"]
                    selected_imputation = st.selectbox("Imputation Technique", imputation_options)

                    if st.button("Apply Preprocessing"):
                        import time
                        my_bar = st.progress(0)
                        for percent_complete in range(100):
                            time.sleep(0.01)
                            my_bar.progress(percent_complete + 1)

                        from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder, LabelEncoder
                        from sklearn.impute import SimpleImputer
                        import pandas as pd

                        df_processed = df.copy()

                        if selected_scaling == "StandardScaler":
                            scaler = StandardScaler()
                            df_processed = pd.DataFrame(scaler.fit_transform(df_processed), columns=df_processed.columns)
                        elif selected_scaling == "MinMaxScaler":
                            scaler = MinMaxScaler()
                            df_processed = pd.DataFrame(scaler.fit_transform(df_processed), columns=df_processed.columns)
                        elif selected_scaling == "RobustScaler":
                            scaler = RobustScaler()
                            df_processed = pd.DataFrame(scaler.fit_transform(df_processed), columns=df_processed.columns)

                        if selected_encoding == "OneHotEncoder":
                            encoder = OneHotEncoder()
                            df_processed = pd.DataFrame(encoder.fit_transform(df_processed).toarray())
                        elif selected_encoding == "LabelEncoder":
                            for col in df_processed.columns:
                                if df_processed[col].dtype == 'object':
                                    le = LabelEncoder()
                                    df_processed[col] = le.fit_transform(df_processed[col])

                        if selected_imputation == "SimpleImputer":
                            imputer = SimpleImputer(strategy='mean')
                            df_processed = pd.DataFrame(imputer.fit_transform(df_processed), columns=df_processed.columns)

                        st.dataframe(df_processed)

                        csv = df_processed.to_csv(index=False)

                        st.download_button(
                            label="Download preprocessed data as CSV",
                            data=csv,
                            file_name='preprocessed_data.csv',
                            mime='text/csv',
                        )
                        st.write("Preprocessing applied!")

                    st.write("Dataset uploaded and ready for preprocessing.")
            else:
                cols = st.columns([1, 2])  # Adjust column widths as needed

                with cols[0]:
                    pass

                with cols[1]:
                    if project_details.get("performance"):
                        st.write(f"**Model Performance:** {project_details['performance']}")

            st.markdown(f"</div>", unsafe_allow_html=True)

# --- EDUCATION ---
with tabs[5]:
    st.subheader("Education")
    st.image("education.jpeg", width=200)
    st.write("**B.Tech in Petroleum Technology Engineering** - Aditya Engineering College (2015–2019) | 75%")

# --- CONTACT FORM ---
with tabs[6]:
    st.subheader("Contact Me")
    st.image("contact us.jpeg", width=200)
    with st.form(key="contact_form"):
        name = st.text_input("Your Name")
        email = st.text_input("Your Email")
        message = st.text_area("Message")
        submit = st.form_submit_button("Send Message")
        if submit:
            st.success("Thank you! Your message has been received.")

# --- FOOTER ---
st.markdown("""<hr style='border: 1px solid #ddd;'>""", unsafe_allow_html=True)
st.markdown("<center>© 2025 Narasimha Naidu Nagam</center>", unsafe_allow_html=True)
