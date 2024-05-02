import streamlit as st
import json
from langchain_community.vectorstores import Qdrant
from langchain_community.document_loaders import JSONLoader
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
import asyncio


def map_columns_to_source(data_model_json, source_json, source_path):
	"""
	Map columns from the data model JSON to corresponding columns in the source JSON based on similar descriptions.

	Parameters:
		data_model_json (dict): JSON data containing the data model columns.
		source_json (dict): JSON data containing the source columns.
		source_path (str): Path to the source file.

	Returns:
		dict: A dictionary mapping data model column names to corresponding source column names.
	"""
	names_map = {}

	for column in data_model_json['columns']:
		column_name_data_model = column['name']
		column_description_data_model = column['description']
		similar_column_description_in_source = asyncio.run(
			find_similar_description(column_description_data_model, source_path))
		name_similar_description_in_source = find_name_by_description(similar_column_description_in_source,
		                                                              source_json['columns'])
		names_map[column_name_data_model] = name_similar_description_in_source

	return names_map


async def find_similar_description(description, json_path):
	model_name = "BAAI/bge-small-en"
	model_kwargs = {"device": "cpu"}
	encode_kwargs = {"normalize_embeddings": True}

	embeddings = HuggingFaceBgeEmbeddings(
		model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs)

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
	description_result = await db_descriptions.asimilarity_search(description)

	similar_description = description_result[0].page_content if description_result else None

	return similar_description


def find_description_by_name(name, name_description_list):
	"""
	Find the description based on its name in a list of dictionaries.

	Parameters:
		name (str): The name to search for.
		name_description_list (list): A list of dictionaries where each dictionary contains "name" and "description" keys.

	Returns:
		str or None: The description corresponding to the given name, or None if not found.
	"""
	for item in name_description_list:
		if item['name'] == name:
			return item['description']
	return None


def find_name_by_description(description, name_description_list):
	"""
	Find the name based on its description in a list of dictionaries.

	Parameters:
		description (str): The description to search for.
		name_description_list (list): A list of dictionaries where each dictionary contains "name" and "description" keys.

	Returns:
		str or None: The name corresponding to the given description, or None if not found.
	"""
	for item in name_description_list:
		if item['description'] == description:
			return item['name']
	return None


def save_uploaded_file(uploaded_file):
	file_path = "./uploaded_json.json"
	with open(file_path, "wb") as f:
		f.write(uploaded_file.getbuffer())
	return file_path


def main():
	st.title("Find Similar Name and Description from JSON")

	uploaded_source = st.file_uploader("Source schema", type=["json"])
	uploaded_data_model = st.file_uploader("Data model schema", type=["json"])
	if uploaded_source and uploaded_data_model:
		source_path = save_uploaded_file(uploaded_source)
		data_model_json = json.load(uploaded_data_model)
		source_json = json.load(uploaded_source)
	if 'checkbox_states' not in st.session_state:
		st.session_state.checkbox_states = {}

	if 'submit' not in st.session_state:
		st.session_state.submit = False

	if st.button("Do the mapping"):
		with st.spinner("Doing the map"):
			st.session_state.names_map = map_columns_to_source(data_model_json, source_json, source_path)
			st.session_state.checkbox_states = {}
	col1, col2, col3 = st.columns([0.4, 0.4, 0.2])
	col1.write("Data Model")
	col2.write("Source")
	col3.write("Valid")
	form = col3.form("my_form")
	for column_name_data_model, column_name_source in st.session_state.names_map.items():
		column_description_data_model = find_description_by_name(column_name_data_model, data_model_json['columns'])
		column_description_source = find_description_by_name(column_name_source, source_json['columns'])
		col1.markdown(column_name_data_model, help=column_description_data_model)
		col2.markdown(column_name_source, help=column_description_source)
		checkbox_state = form.checkbox(" ", key=column_name_data_model)
		st.session_state.checkbox_states[column_name_data_model] = checkbox_state

	if form.form_submit_button("Submit"):
		st.session_state.submit = True

	if st.session_state.submit:
		not_valid_options = [k for k, v in st.session_state.checkbox_states.items() if not v]
		options_in_source = [st.session_state.names_map[name] for name in not_valid_options]
		if len(not_valid_options) > 0:
			option_data_model = not_valid_options[0]
			option_description_data_model = find_description_by_name(option_data_model, data_model_json['columns'])
			source_map_option = st.selectbox(options=options_in_source,
			             label=f'Choose the map for {option_data_model}',
			             placeholder="Select items", index=None,
			             help=option_description_data_model)
			if source_map_option:
				st.session_state.names_map[option_data_model] = source_map_option
		else:
			st.write("Data model - source successfully mapped. Ready to fill the data model.")

if __name__ == "__main__":
	main()
