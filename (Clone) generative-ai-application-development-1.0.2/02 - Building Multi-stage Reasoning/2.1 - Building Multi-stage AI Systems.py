# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning">
# MAGIC </div>
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # Building Multi-stage AI Systems in Databricks
# MAGIC
# MAGIC In this demo we will start building a multi-stage reasoning system using Databricks' features and LangChain. Before we build the chain, first, we will show various components that are commonly used in multi-stage chaining system. 
# MAGIC
# MAGIC In the main section of the demo, we will build a multi-stage system. First, we will build a chain that will answer user questions using DBRX model. The second chain will search for DAIS-2023 talks and will try to find the corresponding video on YouTube. The final, complete chain will recommend videos to the user.
# MAGIC
# MAGIC **Learning Objectives:**
# MAGIC
# MAGIC *By the end of this demo, you will be able to;*
# MAGIC
# MAGIC * Identify that LangChain can include stages/tasks that are not LLMs.
# MAGIC
# MAGIC * Create basic LLM chains to connect prompts and LLMs.
# MAGIC
# MAGIC * Use tools to complete various tasks in the complete system.
# MAGIC
# MAGIC * Construct sequential chains of multiple LLMChains to perform multi-stage reasoning analysis.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Requirements
# MAGIC
# MAGIC Please review the following requirements before starting the lesson:
# MAGIC
# MAGIC * To run this notebook, you need to use one of the following Databricks runtime(s): **14.3.x-cpu-ml-scala2.12 14.3.x-scala2.12**
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Classroom Setup
# MAGIC
# MAGIC Before starting the demo, run the provided classroom setup script. This script will define configuration variables necessary for the demo. Execute the following cell:

# COMMAND ----------

# MAGIC %pip install --upgrade --quiet langchain-core databricks-vectorsearch langchain-community youtube_search wikipedia typing_extensions  
# MAGIC dbutils.library.restartPython() 

# COMMAND ----------

# MAGIC %run ../Includes/Classroom-Setup-03

# COMMAND ----------

# MAGIC %md
# MAGIC **Other Conventions:**
# MAGIC
# MAGIC Throughout this demo, we'll refer to the object `DA`. This object, provided by Databricks Academy, contains variables such as your username, catalog name, schema name, working directory, and dataset locations. Run the code block below to view these details:

# COMMAND ----------

print(f"Username:          {DA.username}")
print(f"Catalog Name:      {DA.catalog_name}")
print(f"Schema Name:       {DA.schema_name}")
print(f"Working Directory: {DA.paths.working_dir}")
print(f"Dataset Location:  {DA.paths.datasets}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Prompt
# MAGIC
# MAGIC Prompt is one of the basic blocks when interacting with GenAI models. They may include instructions, examples and specific context information related to the given task. Let's create a very basic prompt.
# MAGIC

# COMMAND ----------

from langchain.prompts import PromptTemplate

prompt_template = PromptTemplate.from_template("Tell me about a {genre} movie which {actor} is one of the actors.")
prompt_template.format(genre="romance", actor="Brad Pitt")

# COMMAND ----------

# MAGIC %md
# MAGIC ### LLMs
# MAGIC
# MAGIC LLMs are the core component when building compound AI systems. They are the **brain** of the system for reasoning and generating the response.
# MAGIC
# MAGIC Let's see how to interact with **Databricks's DBRX** model.
# MAGIC

# COMMAND ----------

from langchain_community.chat_models import ChatDatabricks

# play with max_tokens to define the length of the response
llm_dbrx = ChatDatabricks(endpoint="databricks-dbrx-instruct", max_tokens = 500)

for chunk in llm_dbrx.stream("Who is Brad Pitt?"):
    print(chunk.content, end="\n", flush=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Retriever
# MAGIC
# MAGIC Retrievers are used when external data is retreived and passed to the model for generating response. There are various types of retrievers such as *document retrievers* and *vector store retrievers*.
# MAGIC
# MAGIC In the next section of the demo, we will use **Databricks Vector Search** as retriever to fetch documents by input query.
# MAGIC
# MAGIC For now, let's try a simple **Wikipedia retriever**.

# COMMAND ----------

from langchain_community.retrievers import WikipediaRetriever
retriever = WikipediaRetriever()
#docs = retriever.get_relevant_documents(query="Brad Pitt")
docs = retriever.invoke(input="Brad Pitt")
print(docs[0])

# COMMAND ----------

# MAGIC %md
# MAGIC ### Tools
# MAGIC
# MAGIC Tools are functions that can be invoked in the chain. Tools has *input parameters* and a *function* to run.
# MAGIC
# MAGIC Here, we have a Youtube search tool. The tool's `description` defines why a tool can be used and the `args` defines what input arguments can be passed to the tool.

# COMMAND ----------

# MAGIC %pip install --upgrade typing_extensions

# COMMAND ----------

# MAGIC %pip install --upgrade typing_extensions==4.7.1

# COMMAND ----------


from langchain_community.tools import YouTubeSearchTool
tool = YouTubeSearchTool()
tool.run("Brad Pitt movie trailer")

# COMMAND ----------

print(tool.description)
print(tool.args)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Chaining
# MAGIC
# MAGIC One of the important features of these components is the ability to **chain** them together. Let's connect the LLM with the prompt.

# COMMAND ----------

print()

# COMMAND ----------

from langchain_community.chat_models import ChatDatabricks

chat_model = ChatDatabricks(
    target_uri="databricks",
    endpoint="databricks-llama-2-70b-chat",
    temperature=0.1,
)

# single input invocation
print(chat_model.invoke("What is MLflow?").content)

# single input invocation with streaming response
for chunk in chat_model.stream("What is MLflow?"):
    print(chunk.content, end="|")

# COMMAND ----------


from langchain_core.output_parsers import StrOutputParser

chain = prompt_template | llm_dbrx | StrOutputParser()

print(chain.invoke({"genre": "romance", "actor": "Brad Pitt"}))

# COMMAND ----------

from langchain_core.output_parsers import StrOutputParser

chain = prompt_template | llm_dbrx | StrOutputParser()
print(chain.invoke({"genre":"romance", "actor":"Brad Pitt"}))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Build a Multi-stage Chain

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create a Vector Store
# MAGIC
# MAGIC **🚨IMPORTANT: Vector Search endpoints must be created before running the rest of the demo. Endpoint names should be in this format; `vs_endpoint_x`. The endpoint will be assigned by username.**

# COMMAND ----------

# assign vs search endpoint by username
vs_endpoint_prefix = "vs_endpoint_"
vs_endpoint_fallback = "vs_endpoint_fallback"
vs_endpoint_name = vs_endpoint_prefix+str(get_fixed_integer(DA.unique_name("_")))
print(f"Vector Endpoint name: {vs_endpoint_name}. In case of any issues, replace variable `vs_endpoint_name` with `vs_endpoint_fallback` in demos and labs.")

# COMMAND ----------

from pyspark.sql import functions as F

vs_index_table_fullname = f"{DA.catalog_name}.{DA.schema_name}.dais_embeddings"
source_table_fullname = f"{DA.catalog_name}.{DA.schema_name}.dais_text"

# load dataset and compute embeddings
df = spark.read.parquet(f"{DA.paths.datasets}/dais/dais23_talks.parquet")
df = df.withColumn("id", F.monotonically_increasing_id())
#df = df.withColumn("embedding", get_embedding("Abstract"))
df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(source_table_fullname)

spark.sql(f"ALTER TABLE {source_table_fullname} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")

# store embeddings in vector store
create_vs_index(vs_endpoint_name, vs_index_table_fullname, source_table_fullname, "Title")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Build First Chain
# MAGIC
# MAGIC The **first chain** will be a simple question-answer prompt using **DBRX**. This chain consist of a `prompt template`, `llm model` and `output parser`.

# COMMAND ----------

from langchain_community.chat_models import ChatDatabricks
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools import YouTubeSearchTool
from databricks.vector_search.client import VectorSearchClient
from langchain.schema.runnable import RunnablePassthrough

llm_dbrx = ChatDatabricks(endpoint=vs_endpoint_name, max_tokens = 1000)
tool_yt = YouTubeSearchTool()

prompt_template_1 = PromptTemplate.from_template(
    """You are a Databricks expert. You will get questions about Databricks. Try to give simple answers and be professional.

    Question: {question}

    Answer:
    """
)

chain1 = ({"question": RunnablePassthrough()} | prompt_template_1 | llm_dbrx | StrOutputParser())
print(chain1.invoke({"question":"How machine learning models are stored in Unity Catalog?"}))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Build Second Chain
# MAGIC
# MAGIC The **second chain** will be used for listing videos relevant to the user's question. In order to get videos, first, we need to search for the DAIS-2023 talks that are already stored in a Vector Search index. After retrieving the relevant titles, we will use YouTube search tool to get the videos for the talks. In the final stage, these videos are passed to the chaing to generate a response for the user.
# MAGIC
# MAGIC This chain consist of a `prompt template`, `llm model` and `output parser`.

# COMMAND ----------

from langchain_community.vectorstores import DatabricksVectorSearch
vsc = VectorSearchClient()
dais_index = vsc.get_index(vs_endpoint_name, vs_index_table_fullname)
query = "how do I use DatabricksSQL"

dvs_delta_sync = DatabricksVectorSearch(dais_index)
docs = dvs_delta_sync.similarity_search(query)

videos = tool_yt.run(docs[0].page_content)

prompt_template_2 = PromptTemplate.from_template(
    """You will get a list of videos related to the user's question which are recorded in DAIS-2023. Encourage the user to watch the videos. List videos with their YouTube links.

    List of videos: {videos}
    """
)
chain2 = ({"videos": RunnablePassthrough()} | prompt_template_2 |  llm_dbrx | StrOutputParser())

# COMMAND ----------

# MAGIC %md
# MAGIC ### Chaining Chains ⛓️
# MAGIC
# MAGIC So far we create chains for each stage. To build a multi-stage system, we need to link these chains together and build a multi-chain system.

# COMMAND ----------

from langchain.schema.runnable import RunnablePassthrough
from operator import itemgetter

multi_chain = ({
    "c":chain1,
    "d": chain2
}| RunnablePassthrough.assign(d=chain2))

multi_chain.invoke({"question":"How machine learning models are stored in Unity Catalog?", "videos":videos})

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Clean up Classroom
# MAGIC
# MAGIC Run the following cell to remove lessons-specific assets created during this lesson.

# COMMAND ----------

DA.cleanup()

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Conclusion
# MAGIC
# MAGIC In this demo, we explored building a multi-stage reasoning system with Databricks' tools and LangChain. We began by introducing common system components and then focused on creating chains for specific tasks like answering user queries and finding DAIS-2023 talks. By the end, participants learned to use LangChain beyond just LLMs and construct sequential chains for multi-stage analyses.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC &copy; 2024 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the 
# MAGIC <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/><a href="https://databricks.com/privacy-policy">Privacy Policy</a> | 
# MAGIC <a href="https://databricks.com/terms-of-use">Terms of Use</a> | 
# MAGIC <a href="https://help.databricks.com/">Support</a>
