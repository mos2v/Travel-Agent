import pandas as pd
from pathlib import Path
from langchain_nvidia_ai_endpoints import ChatNVIDIA
import json
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from typing import List, Optional
from pydantic import BaseModel, Field
from tqdm import tqdm
import argparse
import time
import pickle
from langchain.callbacks.base import BaseCallbackHandler
from langchain_mistralai import ChatMistralAI
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI