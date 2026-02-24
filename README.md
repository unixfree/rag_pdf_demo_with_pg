### rag_pdf_demo_with_pg
PostgreSQL(EPAS) 의 pgvector 를 이용한 pdf 파일 기반 RAG 코드입니다.
OpenAI의 text-embedding-3-small 와 gpt-4o-mini 를 사용하였습니다.


#### 테스트 방법
0. python 환경 만들기
```
python3.12 -m venv venv

# 가상환경 활성화
source venv/bin/activate

# 버전 확인 (이제 python만 쳐도 3.12로 실행됩니다)
python --version

pip install streamlit langchain langchain-openai langchain-community langchain-postgres pypdf psycopg2-binary python-dotenv
```

1. git clone
```
git clone
```

2. 환경 변수 적용
```
cp env .env
vi .env
# OpenAI API Key
OPENAI_API_KEY=<OpenAI API Key 적용>
# PostgreSQL Connection String (using psycopg2)
DATABASE_URL=postgresql+psycopg2://enterprisedb:enterprisedb@localhost:5444/enterprisedb
```

3. 수행
```
streamlit run app.py
```





