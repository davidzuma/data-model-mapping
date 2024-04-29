import streamlit as st
import json
from langchain_community.vectorstores import Qdrant
from langchain_community.document_loaders import JSONLoader
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
import asyncio


async def find_similar_name_and_description(name, description, json_path):
	model_name = "BAAI/bge-small-en"
	model_kwargs = {"device": "cpu"}
	encode_kwargs = {"normalize_embeddings": True}

	embeddings = HuggingFaceBgeEmbeddings(
		model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs)

	loader_names = JSONLoader(
		file_path=json_path,
		jq_schema='.columns[].name',
		text_content=False
	)
	names = loader_names.load()
	db_names = Qdrant.from_documents(
		names,
		embeddings,
		location=":memory:",  # Local mode with in-memory storage only
		collection_name="column names")

	loader_descriptions = JSONLoader(
		file_path=json_path,
		jq_schema='.columns[].description',
		text_content=False
	)
	descriptions = loader_descriptions.load()

	db_descriptions = Qdrant.from_documents(
		descriptions,
		embeddings,
		location=":memory:",  # Local mode with in-memory storage only
		collection_name="description")

	name_result = await db_names.asimilarity_search(name)
	description_result = await db_descriptions.asimilarity_search(description)

	similar_name = name_result[0].page_content if name_result else None
	similar_description = description_result[0].page_content if description_result else None

	return similar_name, similar_description


def save_uploaded_file(uploaded_file):
	file_path = "./uploaded_json.json"
	with open(file_path, "wb") as f:
		f.write(uploaded_file.getbuffer())
	return file_path


def main():
	st.title("Find Similar Name and Description from JSON")

	# File uploader for first JSON file
	uploaded_file1 = st.file_uploader("Upload first JSON file", type=["json"])

	# File uploader for second JSON file
	uploaded_file2 = st.file_uploader("Upload second JSON file", type=["json"])

	if uploaded_file1 is not None and uploaded_file2 is not None:
		json_path1 = save_uploaded_file(uploaded_file1)
		# json_path2 = save_uploaded_file(uploaded_file2)
		st.success("Files successfully uploaded and saved to disk.")
		json_data2 = json.load(uploaded_file2)
		col1, col2, col3 = st.columns([0.4, 0.4, 0.2])
		col1.write("Source Data")
		col2.write("Data Model")
		col3.write("Valid")
		for i, column in enumerate(json_data2['columns']):
			column_name = column['name']
			column_description = column['description']
			similar_name, similar_description = asyncio.run(
				find_similar_name_and_description(column_name, column_description, json_path1))
			col1.markdown(column_name, help = column_description)
			col2.markdown(similar_name if similar_name else "-", help = similar_description)
			col3.checkbox("", key=i)


if __name__ == "__main__":
	main()
