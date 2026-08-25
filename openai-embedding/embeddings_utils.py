import textwrap as tr
from typing import List, Optional

import matplotlib.pyplot as plt
import plotly.express as px
from scipy import spatial
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import average_precision_score, precision_recall_curve

from openai import OpenAI
import numpy as np
import pandas as pd

# OpenAI 클라이언트 초기화 (API 실패 시 최대 5회 재시도)
client = OpenAI(max_retries=5)


def get_embedding(text: str, model="text-embedding-3-small", **kwargs) -> List[float]:
    """단일 텍스트의 임베딩 벡터를 반환하는 함수"""
    
    # 줄바꿈 문자를 공백으로 대체 (성능 저하 방지)
    text = text.replace("\n", " ")

    # OpenAI 임베딩 API 호출
    response = client.embeddings.create(input=[text], model=model, **kwargs)

    # 첫 번째 결과의 임베딩 벡터 반환
    return response.data[0].embedding


def get_embeddings(
    list_of_text: List[str], model="text-embedding-3-small", **kwargs
) -> List[List[float]]:
    """여러 텍스트의 임베딩 벡터 리스트를 반환하는 함수"""
    
    # 배치 크기 제한 검사 (OpenAI API 최대 허용치: 2048)
    assert len(list_of_text) <= 2048, "The batch size should not be larger than 2048."

    # 모든 텍스트의 줄바꿈 문자를 공백으로 대체 (성능 저하 방지)
    list_of_text = [text.replace("\n", " ") for text in list_of_text]

    # OpenAI 임베딩 API 일괄 호출
    data = client.embeddings.create(input=list_of_text, model=model, **kwargs).data
    
    # 각 텍스트의 임베딩 벡터만 추출하여 리스트로 반환
    return [d.embedding for d in data]