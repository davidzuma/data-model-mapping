from langchain_community.vectorstores import Qdrant
from langchain_community.document_loaders import JSONLoader
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
import asyncio
import json

model_name = "BAAI/bge-small-en"
model_kwargs = {"device": "cpu"}
encode_kwargs = {"normalize_embeddings": True}
embeddings = HuggingFaceBgeEmbeddings(
    model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
)



# loader = JSONLoader(
#     file_path='./data/table_1.json',
#     jq_schema='.columns',
#     text_content= False)
#
# data = loader.load()
# print(data)
# text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
# documents = text_splitter.split_documents(raw_documents)
# db = Chroma.from_documents(documents, OpenAIEmbeddings())
#
# db = Qdrant.afrom_documents(documents, embeddings, "http://localhost:6333")

if __name__ == '__main__':
    model_name = "BAAI/bge-small-en"
    model_kwargs = {"device": "cpu"}
    encode_kwargs = {"normalize_embeddings": True}

    embeddings = HuggingFaceBgeEmbeddings(
        model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs)

    loader_names = JSONLoader(
        file_path='./data/table_1.json',
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
        file_path='./data/table_1.json',
        jq_schema='.columns[].description',
        text_content=False
        )
    descriptions = loader_descriptions.load()

    db_descriptions = Qdrant.from_documents(
        descriptions,
        embeddings,
        location=":memory:",  # Local mode with in-memory storage only
        collection_name="description")

    with open('./data/table_2.json') as f:
        data = json.load(f)
    for column in data['columns']:
        query_name = column['name']
        query_description = column['description']

        name = asyncio.run(db_names.asimilarity_search(query_name))[0].page_content
        description = asyncio.run(db_descriptions.asimilarity_search(query_description))[0].page_content
        print("input_name:", query_name, "input_description:", query_description)
        print("name:", name, "description:", description)




    # url = "http://localhost:6333"
    # qdrant = Qdrant.from_documents(
    #     documents,
    #     embeddings,
    #     url=url,
    #     prefer_grpc=True,
    #     collection_name="my_documents",
    # )

