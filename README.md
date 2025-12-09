# Cross_Domain_Knowledge_Mapping
This project focuses on mapping and analyzing knowledge across multiple domains using a curated NLP dataset. It integrates data from ten diverse fields to create a unified knowledge graph for better insights and exploration.

## Dataset

The dataset used, `cross_domain_nlp_dataset_5000.csv`, contains around 5000 rows and 4 columns. It covers ten domains: Sociology, Computer Science, Linguistics, Mathematics, Economics, Psychology, Physics, Chemistry, Environmental Science, and Biology. Choosing rich and clean datasets like those from Kaggle (e.g., Reuters) is essential for effective NLP model training and evaluation.

## Libraries Used

This project relies on several Python libraries for data processing, NLP, and visualization and UI:
pip install \
  pandas \
  numpy \
  spacy \
  nltk \
  scikit-learn \
  networkx \
  matplotlib \
  seaborn \
  plotly \
  pyvis \
  streamlit \
  Flask \
  flask-cors \
  sentence-transformers \
  transformers \
  torch \
  textblob \
  pyarrow \
  pydeck \
  altair \
  requests \
  GitPython \
  fuzzywuzzy

python -m spacy download en_core_web_sm


## Features

- Cross-domain NLP dataset integration
- Knowledge graph creation and visualization
- Support for 10 diverse academic and scientific domains
- Secure user authentication with JWT-based login and token verification
- User registration and password reset via RESTful API endpoints
- Role-aware user management using a user type field (e.g., admin, researcher, student)
- Separate frontend routes for login, registration, and forgot password pages

## Usage

### 1. Backend (Flask API)
cd ui
python server.py

text

- The authentication API and HTML pages will be available at `http://127.0.0.1:5000/`.
- Open `http://127.0.0.1:5000/` in your browser to access the login page.

### 2. Frontend (Streamlit app)

cd ui
streamlit run app.py

text

- Streamlit will print a local URL (e.g., `http://localhost:8501`) in the terminal.
- Open that URL in your browser to use the cross-domain knowledge mapping interface.


## License

This project is licensed under the MIT License.  
See the [LICENSE](LICENSE) file for details.

