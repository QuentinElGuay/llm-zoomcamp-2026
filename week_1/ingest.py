import logging

import requests


logger = logging.getLogger(__name__)


def load_course_data():
    from gitsource import GithubRepositoryDataReader, chunk_documents

    reader = GithubRepositoryDataReader(
        repo_owner="DataTalksClub",
        repo_name="llm-zoomcamp",
        commit_id="8c1834d",
        allowed_extensions={"md"},
        filename_filter=lambda path: "/lessons/" in path
    )

    files = reader.read()
    
    documents = []

    for file in files:
        doc = file.parse()
        documents.append(doc)

    chunks = chunk_documents(documents, size=2000, step=1000)

    return chunks

def load_faq_data():
    docs_url = "https://datatalks.club/faq/json/courses.json"
    response = requests.get(docs_url)
    courses_raw = response.json()

    documents = []
    url_prefix = "https://datatalks.club/faq"

    for course in courses_raw:
        course_url = f"""{url_prefix}{course["path"]}"""
        course_response = requests.get(course_url)
        course_response.raise_for_status()
        course_data = course_response.json()

        documents.extend(course_data)

    return documents

def build_index(index, documents):
    index.fit(documents)

    logger.debug('Index built with %s documents', len(documents))

    return index
