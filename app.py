# Core Pkg
import streamlit as st
import streamlit.components.v1 as stc

# Load EDA 
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load Our Dataset
def load_data(data):
    df = pd.read_csv("Coursera.csv")
    return df



# Fxn
# Vectorize with TF-IDF + Cosine Similarity Matrix
def vectorize_text_to_cosine_mat(data):
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(data)
    cosine_sim_mat = cosine_similarity(tfidf_matrix)
    return cosine_sim_mat

# Recommendation Sys
def get_recommendation(title, cosine_sim_mat, df, num_of_rec=10):
    # Indices of the course
    course_indices = pd.Series(df.index, index=df['Course Name']).drop_duplicates()
    # Index of courses
    idx = course_indices[title]

    # Look into the cosine matrix for that index
    sim_scores = list(enumerate(cosine_sim_mat[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    selected_course_indices = [i[0] for i in sim_scores[1:]]
    selected_course_scores = [i[1] for i in sim_scores[1:]]

    # Get the dataframe & title
    result_df = df.iloc[selected_course_indices]
    result_df['similarity_score'] = selected_course_scores
    final_recommended_courses = result_df[['Course Name', 'similarity_score', 'Course URL', 'Difficulty Level', 'Skills', 'Course Description']]
    return final_recommended_courses.head(num_of_rec)

# Search for a course
def search_term_if_not_found(term, df):
    term_lower = term.lower()
    result_df = df[df['Course Name'].str.lower().str.contains(term_lower) | df['Course Description'].str.lower().str.contains(term_lower)]
    return result_df

# CSS Style
RESULT_TEMP = """
<div style="width: 90%; margin: 10px auto; padding: 15px; border-radius: 10px; box-shadow: 0 0 15px 5px #ccc; background-color: #f8f9fa;">
    <h4 style="color: #343a40; font-size: 1.8em; margin-bottom: 10px; text-align: center; text-transform: uppercase;">{}</h4>
    <p style="color: #007bff; font-size: 1.2em; margin-bottom: 5px;"><span style="color: #000; font-weight: bold;">📈 Score:</span> {}</p>
    <p style="color: #007bff; font-size: 1.2em; margin-bottom: 5px;"><span style="color: #000; font-weight: bold;">🔗 URL for Description:</span> <a href="{}" target="_blank" style="text-decoration: none; color: #0062cc;">Link</a></p>
    <p style="color: #007bff; font-size: 1.2em; margin-bottom: 5px;"><span style="color: #000; font-weight: bold;">👨 Difficulty Level:</span> {}</p>
    <p style="color: #007bff; font-size: 1.2em; margin-bottom: 5px;"><span style="color: #000; font-weight: bold;">🧑‍🎓🏽‍🎓 Skills:</span> {}</p>
</div>
"""

def main():
    st.title("🚀 Course Recommendation App ")
    st.write(
        """
        <div style="text-align: center; font-size: 1.5em; color: #007bff; margin-bottom: 20px; text-transform: uppercase; font-weight: bold;">
            Discover the best courses for your learning journey!
        </div>
        """,
        unsafe_allow_html=True
    )

    menu = ["Home", "Recommend", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    df = load_data("data/Coursera.csv")
    if choice == "Home":
        st.subheader("🏠 Home")
        st.dataframe(df.head(10))

    elif choice == "Recommend":
        st.subheader("🎓 Recommend Courses")
        cosine_sim_mat = vectorize_text_to_cosine_mat(df['Course Name'] + ' ' + df['Course Description'])
        search_term = st.text_input("Search")
        num_of_rec = st.sidebar.number_input("Number", 4, 30, 7)
        if st.button("🔍 Recommend"):
            if search_term:
                try:
                    results = get_recommendation(search_term, cosine_sim_mat, df, num_of_rec)
                    with st.expander("Results as JSON"):
                        st.json(results.to_dict(orient='records'))

                    if not results.empty:
                        for row in results.iterrows():
                            rec_title = row[1]['Course Name']
                            rec_score = row[1]['similarity_score']
                            rec_url = row[1]['Course URL']
                            rec_level = row[1]['Difficulty Level']
                            rec_skills = row[1]['Skills']
                            rec_description = row[1]['Course Description']

                            stc.html(RESULT_TEMP.format(rec_title, rec_score, rec_url, rec_level, rec_skills, rec_description), height=380)
                except Exception as e:
                    st.error(f"Not Found: {e}")
                    st.warning("Suggested Options Include:")
                    result_df = search_term_if_not_found(search_term, df)
                    st.dataframe(result_df)
            else:
                st.warning("Please enter a search term.")
    else:
        st.subheader("📘 About")
        st.markdown(
            """
            Built with ❤️ using Streamlit and Pandas by:

- **EL AZZOUZY Nouralhouda**
- **IMSEG Safaa**

As part of the UEX Project at ENSAM Meknes, December 2023.

            """
        )

if __name__ == '__main__':
    main()

